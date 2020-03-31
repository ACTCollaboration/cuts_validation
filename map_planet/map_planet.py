#!/usr/bin/env python
'''
Slightly modified version of tenki/planet_map.py that produces total intensity
planet maps maps with no absolute calibration applied.
'''
from __future__ import print_function, division
import numpy as np, sys, os, errno
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, pmat, mpi, errors, gapfill, enmap, bench, enplot
from enlib import fft, array_ops, scanutils
from enact import filedb, actscan, actdata, cuts, nmat_measure

opj = os.path.join

config.set("pmat_cut_type",  "full")

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist", type=float, default=0.2)
parser.add_argument("-e", "--equator", action="store_true")
parser.add_argument("--sim", type=str, default=None,
                    help="Passing a sel here sets up simulation mode. "
                    "The simulations will consist of data from the sim "
                    "sel TODs with the scanning pattern of the real TODs, "
                    "and with the signal read off from the area map")
parser.add_argument("--noiseless", action="store_true",
                    help="Replace signal with simulation instead of adding them."
                    "This can be used to get noise free transfer functions")
parser.add_argument("--dbox", type=str, default=None,
                    help="Select only detectors in y1:y2,x1:x2 in the focalplane,"
                    "relative to the center of the array, in degrees.")
parser.add_argument("--tags", type=str, default=None)
args = parser.parse_args()

comm = mpi.COMM_WORLD
filedb.init()
ids  = filedb.scans[args.sel]
R    = args.dist * utils.degree
csize= 100

dtype= np.float64
area = enmap.read_map(args.area).astype(dtype)
ncomp= 1 # We only need I.
shape= area.shape[-2:]
model_fknee = 10
model_alpha = 10
sys = "hor:"+args.planet
if args.equator: sys += "/0_0"
utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag:  prefix += args.tag + "_"
if args.dbox: dbox = np.array([[float(w) for w in tok.split(":")] \
                               for tok in args.dbox.split(",")]).T*utils.degree
else: dbox = None

if args.sim:
    sim_ids = filedb.scans[args.sim][:len(ids)]
    if area.ndim == 2:
        tmp = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
        tmp[0] = area
        area = tmp

def smooth(tod, srate):
    ft   = fft.rfft(tod)
    freq = fft.rfftfreq(tod.shape[-1])*srate
    flt  = 1/(1+(freq/model_fknee)**model_alpha)
    ft  *= flt
    fft.ifft(ft, tod, normalize=True)
    return tod

abscal_dict_sub = {} # For abscal values on this process.

for ind in range(comm.rank, len(ids), comm.size):
    id    = ids[ind]
    bid   = id.replace(":","_")
    entry = filedb.data[id]
    if args.tags: entry.tag = args.tags

    # Read the tod as usual
    try:
        if not args.sim:
            with bench.show("read"):
                d = actdata.read(entry)
        else:
            sim_id    = sim_ids[ind]
            sim_entry = filedb.data[sim_id]
            with bench.show("read"):
                d  = actdata.read(entry, ["boresight"])
                d += actdata.read(sim_entry, exclude=["boresight"])

        # Store the abscal value.
        abscal_dict_sub[id] = d.gain_correction[entry.tag]
        abscal = d.gain_correction[entry.tag]

        if d.gain_mode == 'mce':
            abscal /= d.mce_gain
        elif d.gain_mode == 'mce_compat':
            abscal /= d.mce_gain * 1217.8583043
        else:
            raise ValueError('gain_mode {} not understood'.format(d.gain_mode))

        with bench.show("calibrate"):
            d = actdata.calibrate(d, exclude=["autocut"])

        rel_gain = d.gain_raw.copy() # To store later on.

        if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
        # Select detectors if needed.
        if dbox is not None:
            mid  = np.mean(utils.minmax(d.point_template, 0), 0)
            off  = d.point_template-mid
            good = np.all((off > dbox[0])&(off < dbox[1]),-1)
            d    = d.restrict(dets=d.dets[good])
    except errors.DataMissing as e:
        print("Skipping %s (%s)" % (id, e.message))
        continue
    print("Processing %s" % id, d.ndet, d.nsamp)

    # Very simple white noise model.
    with bench.show("ivar"):
        tod  = d.tod
        del d.tod

        tod -= np.mean(tod,1)[:,None]
        tod  = tod.astype(dtype)
        diff = tod[:,1:]-tod[:,:-1]
        diff = diff[:,:diff.shape[-1]//csize*csize].reshape(d.ndet,-1,csize)
        ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5) # Per detector.

        # Save var for later.
        var_per_det = 1/ivar.copy()
        dets = d.dets.copy()

        del diff

    # Estimate noise level.
    asens = np.sum(ivar)**-0.5 / d.srate**0.5
    #print('asens', asens)
    with bench.show("actscan"):
        scan = actscan.ACTScan(entry, d=d)
    with bench.show("pmat"):
        pmap = pmat.PmatMap(scan, area, sys=sys)
        pcut = pmat.PmatCut(scan)
        rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
        div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
        hits = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
        junk = np.zeros(pcut.njunk, dtype)
    # Generate planet cut.
    with bench.show("planet cut"):
        planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
                                        args.planet, R)

    if args.sim:
        if args.noiseless: tod_orig = tod.copy()
        with bench.show("inject"):
            pmap.forward(tod, area)
    # Compute atmospheric model
    with bench.show("atm model"):
        model  = smooth(gapfill.gapfill_joneig(tod, planet_cut, inplace=False), d.srate)
    if args.sim and args.noiseless:
        model -= smooth(gapfill.gapfill_joneig(tod_orig, planet_cut, inplace=False), d.srate)
        tod   -= tod_orig
        del tod_orig
    with bench.show("atm subtract"):
        tod -= model
        del model
        tod  = tod.astype(dtype, copy=False)
    # Should now be reasonably clean of correlated noise.
    # Proceed to make simple binned map
    with bench.show("rhs"):
        tod *= ivar[:,None]
        pcut.backward(tod, junk)
        pmap.backward(tod, rhs)

    with bench.show("hits"):
        for i in range(ncomp):
            div[i,i] = 1
            pmap.forward(tod, div[i])   # div -> tod.
            tod_ones = tod.copy()
            tod *= ivar[:,None]
            pcut.backward(tod, junk)    # tod -> junk.
            pcut.backward(tod_ones, junk)    # tod -> junk.
            div[i] = 0
            pmap.backward(tod, div[i])  # tod -> div.
            pmap.backward(tod_ones, hits[i])  # tod -> hits.

    with bench.show("map"):
        idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5, fallback="scalar")
        map  = enmap.map_mul(idiv, rhs)
    # Estimate central amplitude
    c = np.array(map.shape[-2:])/2
    crad  = 50
    mcent = map[:,c[0]-crad:c[0]+crad,c[1]-crad:c[1]+crad]
    mcent = enmap.downgrade(mcent, 4)
    amp   = np.max(mcent)
    print("%s amp %7.3e asens %7.3e" % (id, amp/1e6, asens))
    with bench.show("write"):

        # Store maps in individual directories with id as name.
        outdir = opj(prefix, entry.id)
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        # Make sure map, rhs, div are 2d and undo the abscal.
        map = map[0,...] / abscal
        rhs = rhs[0,...] / abscal
        div = div[0,0,...] / (abscal ** 2)
        hits = hits[0,0,...]

        mdict = {'map' : map, 'rhs' : rhs, 'div' : div, 'hits' : hits}

        for mkey in mdict:
            enmap.write_map(opj(outdir, '{}_{}_I.fits'.format(mkey, entry.tag)), mdict[mkey])

            plot = enplot.plot(mdict[mkey], quantile=0.001, colorbar=True, ticks=.5)
            enplot.write(opj(outdir, '{}_{}_I'.format(mkey, entry.tag)), plot)

        # Save dets and var and rel gain.
        if d.gain_mode == 'mce':
            rcal_unit = 'pW/DAC'
        elif d.gain_mode == 'mce_compat':
            rcal_unit = 'pW/DACf'
        else:
            raise ValueError('gain_mode {} not understood'.format(d.gain_mode))

        with open(opj(outdir, 'det_stats_{}.txt'.format(entry.tag)), 'w') as f:
            f.write('name\tnvar[uK]\trcal[{}]\n'.format(rcal_unit))

            for lidx in xrange(d.ndet):
                f.write('{}\t{}\t{}\n'.format(dets[lidx], var_per_det[lidx], rel_gain[lidx]))

    del d, scan, pmap, pcut, tod, map, rhs, div, idiv, junk

# Gather and save all asbcal values that we undid.
list_of_dicts = comm.gather(abscal_dict_sub, root=0)
if comm.Get_rank() == 0:
    abscal_dict = {}
    for d in list_of_dicts:
        abscal_dict.update(d)

    outfile = opj(prefix, 'abscal_original.txt')
    with open(outfile, 'w') as f:
        for scan_id in ids:
            entry = filedb.data[scan_id]
            try:
                f.write('{} {} {}\n'.format(entry.id, entry.tag, abscal_dict[scan_id]))
            except KeyError:
                f.write('{} {} {}\n'.format(entry.id, entry.tag, np.nan))
