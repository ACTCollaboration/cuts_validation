#!/usr/bin/env python
'''
Produce maps of the cuts and hits for the selected TODs.
Mostly taken from map_cuts.py script by Sigurd.
'''

from __future__ import print_function

import numpy as np
import os
import sys
import pipes
from mpi4py import MPI

from enlib import config, utils, pmat, errors, sampcut
from enact import filedb, actdata, actscan
from pixell import enmap

opj = os.path.join
comm = MPI.COMM_WORLD
dtype = np.float64

def mkoutdir(outdir):
    if comm.Get_rank() == 0:
        utils.mkdir(outdir)

def write_settings(outdir):
    '''Store parameters for this run. '''

    if comm.Get_rank() == 0:
        config.save(opj(outdir, 'config.txt'))

        with open(opj(outdir, 'env.txt'), 'w') as f:
            for k,v in os.environ.items():
                f.write('{}: {}\n'.format(k, v))

        with open(opj(outdir, 'args.txt'), 'w') as f:
            argstring = " ".join([pipes.quote(a) for a in sys.argv[1:]])
            f.write(argstring + "\n")

def write_scans(outdir, filelist):
    '''
    Store scan indices.

    Parameters
    ----------
    outdir : str
        Absolute path to output directory.
    filelist : array-like of str
        Sequence of scan indices (XXXXXXXXXX.XXXXXXXXXX.arX:fXXX).
    '''

    if comm.Get_rank() == 0:
        with open(opj(outdir, 'ids.txt'), 'w') as f:
            for idx in filelist:
                f.write("{}\n".format(idx))

def allocate_output(area):
    '''
    Parameters
    ----------
    area : str
        Absolute path to footprint map.

    Returns
    -------
    hitmap : (1, ny, nx) enmap
    cutmap : (1, ny, nx) enmap
    '''

    shape, wcs = enmap.read_map_geometry(area)
    hitmap = enmap.zeros((1,)+shape[-2:], wcs, dtype)
    cutmap = hitmap * 0

    return hitmap, cutmap

def read_metadata(entry):
    '''
    Parameters
    ----------
    entry : filedb.data object

    Returns
    -------
    data : enlib.dataset.DataSet instance
    '''

    data = actdata.read(entry, exclude=['tod'])
    data = actdata.calibrate(data, exclude=['autocut'])
    if data.ndet == 0 or data.nsamp == 0:
        raise errors.DataMissing("No data in tod")
    return data

def map_cuts(entry, data, hitmap, cutmap, keep_buffer=False,
             nocommon_frac=None):
    '''
    Project hits and hits-cuts onto the sky.

    parameters
    ----------
    entry : filedb.data object
    data : actdata instance
    hitmap : (1, ny, nx) enmap
    cutmap : (1, ny, nx) enmap
    keep_buffer : bool, optional
        Do not remove the 200 sample buffers before and after cuts.
    common_frac : float, optional
        Do not consider cuts that are at least common to this
        fraction of detectors.
    '''

    scan = actscan.ACTScan(entry, d=data)
    pmap = pmat.PmatMap(scan, hitmap)
    cut = data.cut

    if nocommon_frac is not None:
        cut = remove_common(cut, frac=nocommon_frac)

    if keep_buffer is False:
        cut = remove_buffer(cut)

    tod = np.full([data.ndet, data.nsamp], 1.0, dtype)
    pmap.backward(tod, hitmap)
    sampcut.gapfill_const(cut, tod, 0.0, inplace=True)
    pmap.backward(tod, cutmap)

def remove_common(cut, frac=0.5):
    '''
    Remove cuts that are cut by most detectors.

    Parameters
    ----------
    cut : Sampcut instance
    frac : float, optional
        Remove cuts common to at least this fraction of detectors.

    Returns
    -------
    cuts : Sampcut instance
        Copy of input without common cuts.
    '''

    samp_mask = np.sum(cut.to_mask(), 0) > cut.ndet * frac
    all_cut = sampcut.from_mask(samp_mask[None])
    cut = ~(~cut * all_cut)

    return cut

def remove_buffer(cut):
    '''
    Remove the 200 sample buffer before and after each cut.

    Parameters
    ----------
    cut : Sampcut instance

    Returns
    -------
    cut : Sampcut instance
        Copy of input without buffer.
    '''

    n = 200
    ranges = cut.ranges
    # Only select cuts larger than 2n.
    mask = (ranges[:,1] - ranges[:,0]) > 2 * n

    ranges[mask,0] += n
    ranges[mask,1] -= n

    return sampcut.Sampcut(ranges, cut.detmap, cut.nsamp)

def pas_per_season(season):

    if season == 's13':
        return ['pa1']
    elif season == 's14':
        return ['pa1', 'pa2']
    elif season == 's15':
        return ['pa1', 'pa2', 'pa3']
    elif season == 's16':
        return ['pa2', 'pa3', 'pa4']
    elif season in ('s17', 's18', 's19'):
        return ['pa4', 'pa5', 'pa6']
    else:
        raise ValueError('No arrays found for season {}'.
                         format(season))

def freqs_per_pa(pa):

    if pa in ('pa1', 'pa2'):
        return ['f150']
    elif pa in ('pa3', 'pa5', 'pa6'):
        return ['f090', 'f150']
    elif pa == 'pa4':
        return ['f150', 'f220']
    else:
        raise ValueError('No freqs found for season {}'.
                         format(pa))

def split_ids(ids):
    '''
    Split scan indices into arrays, freqs and seasons.

    Returns
    -------
    ids_split : dict of arrays
    '''

    seasons = ['s13', 's14', 's15', 's16', 's17', 's18', 's19']
    ids_split = {}
    for season in seasons:
        for pa in pas_per_season(season):
            for freq in freqs_per_pa(pa):
                key = '{}_{}_{}'.format(pa, freq, season)
                ids_split[key] = []

    for scan_id in ids:
        freq = scan_id[-4:]
        season = str(12 + int(filedb.extractors['season'](scan_id[:-5])))
        pa = filedb.extractors['ar'](scan_id[:-5])
        key = 'pa{}_{}_s{}'.format(pa, freq, season)
        ids_split[key].append(scan_id)

    for key in list(ids_split):
        if ids_split[key] == []:
            ids_split.pop(key)

    return ids_split

def remove_sidelobe_cut(entry):
    '''
    Remove the sidelobe cuts from the data object.

    Parameters
    ----------
    entry : filedb.data object
    '''

    for cut_type in entry.cut['subs']:
        if 'sidelobe_cut' in cut_type:
            entry.cut['subs'].remove(cut_type)

if __name__ == '__main__':

    parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
    parser.add_argument("area")
    parser.add_argument("sel")
    parser.add_argument("odir")
    parser.add_argument("--keep-sidelobe-cut", action="store_true",
                        default=False, dest='keep_sidelobe')
    parser.add_argument("--keep-buffer", action='store_true', default=False,
                        dest='keep_buffer')
    parser.add_argument("--nocommon-fraction", type=float, default=0.5,
                        dest='nocommon')

    args = parser.parse_args()

    filedb.init()
    db = filedb.data

    outdir = args.odir
    mkoutdir(outdir)
    write_settings(outdir)

    filedb.init()
    ids = filedb.scans[args.sel]
    id_dict = split_ids(ids)

    for scan_key, scan_ids in id_dict.items():

        outdir_sub = opj(outdir, scan_key)
        mkoutdir(outdir_sub)
        write_scans(outdir_sub, scan_ids)

        scan_ids_loc = np.array_split(scan_ids, comm.Get_size())[comm.Get_rank()]
        hitmap, cutmap = allocate_output(args.area)

        for idx, scan_id in enumerate(scan_ids_loc):

            print('rank {:3d}: working on {}: {:3d}/{:03d}: {}'.format(
                comm.Get_rank(), scan_key, idx+1, len(scan_ids_loc), scan_id))

            entry = filedb.data[scan_id]

            if args.keep_sidelobe is False:
                remove_sidelobe_cut(entry)

            try:
                data = read_metadata(entry)
            except errors.DataMissing as e:
                print('rank {:3d}: {} skipped ({})'.format(
                    comm.Get_rank(), scan_id, e.message))
                continue

            nocommon = None if args.nocommon == -1 else args.nocommon

            map_cuts(entry, data, hitmap, cutmap,
                     keep_buffer=args.keep_buffer,
                     nocommon_frac=nocommon)

        comm.Barrier()
        hitmap = utils.reduce(hitmap, comm)
        cutmap = utils.reduce(cutmap, comm)

        if comm.Get_rank() == 0:
            # Up to now cutmap is hits - cuts. We want cuts only.
            cutmap = hitmap - cutmap
            enmap.write_map(opj(outdir_sub, 'hits.fits'), hitmap[0])
            enmap.write_map(opj(outdir_sub, 'cuts.fits'), cutmap[0])
