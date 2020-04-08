#!/usr/bin/env python  
'''
Compute the cross correlation between a sky map and the cut (ratio) map. 
'''
import numpy as np
import os
import glob
import argparse
from mpi4py import MPI

from pixell import enmap, utils
import pymaster as nmt
import nawrapper as nw
import nawrapper.maptools as maptools

from plot_cuts import read_signal_map

opj = os.path.join
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_subdirs(basedir, pattern='pa*_f*_s*'):

    files = glob.glob(opj(basedir, pattern))
    subdirs = [os.path.split(f)[1] for f in files]
    
    return subdirs

def get_freq(subdir):
    
    for s in subdir.split('_'):
        if s.startswith('f'):
            return s

def smooth(m):
    '''
    Local smoothing of input map.
    
    Parameters
    ----------
    m : enmap
        Input map

    Returns
    -------
    m_smoothed : enmap
        Output map with same shape as input.
    '''
    
    m_smoothed = enmap.downgrade(m, 2)
    m_smoothed = enmap.project(m_smoothed, m.shape, m.wcs, order=1)
    return m_smoothed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("sdir", 
                        help='Source directory containing "pa*_f*_s*" directories.')
    parser.add_argument("mcmdir", 
                        help='Namaster mode coupling matrix directory')
    parser.add_argument("odir", 
                        help='Output directory')
    parser.add_argument("signal-map", nargs='+',
                        help='Path to .fits file to correlate with cuts. '
                        'Can be list of paths for different freqs or single path with '
                        '"{freq}" in filename.') 
    parser.add_argument("signal-map-scrfree", nargs='+',
                        help='Path to second signal map (e.g. one without point sources). '
                        'Can be list of paths for different freqs or single path with '
                        '"{freq}" in filename.') 
    parser.add_argument("--downgrade", type=float, default=1.,
                        help='Downgrade maps by this factor')
    parser.add_argument("--lmax", type=int, default=15000)
    
    args = parser.parse_args()

    if rank == 0:
        utils.mkdir(args.odir)
        utils.mkdir(args.mcmdir)

    subdirs = get_subdirs(args.sdir)
    for subdir in subdirs[rank::comm.Get_size()]:
        
        outdir = opj(args.odir, subdir)
        utils.mkdir(outdir)

        freq = get_freq(subdir)
        print('[rank {:03d}]: reading maps'.format(rank))
        mask = enmap.read_map(opj(args.sdir, subdir, 'hits.fits'))
        ratio = enmap.read_map(opj(args.sdir, subdir, 'cuts.fits'))        
        
        signal_map = args.signal_map
        if len(signal_map) == 1:
            signal_map = signal_map[0]
        signal = read_signal_map(signal_map, freq, ratio.shape, ratio.wcs)

        print('[rank {:03d}]: processing maps'.format(rank))
        if args.downgrade != 1:
            signal = enmap.downgrade(signal, args.downgrade)
            mask = enmap.downgrade(mask,  args.downgrade)
            ratio = enmap.downgrade(ratio,  args.downgrade)
                
        ratio[mask == 0] *= 0
        ratio[mask != 0] /= mask[mask != 0]

        mask[mask != 0] = 1.

        ratio = smooth(ratio)
        mask = maptools.apod_C2(mask, 0.2)

        namap_cut = nw.namap_car(maps=(ratio, None, None), masks=mask)
        namap_sig = nw.namap_car(maps=(signal, None, None), masks=mask)
        
        bins = nw.create_binning(lmax=args.lmax, lmin=2, widths=50)
        print('[rank {:03d}]: calculating mode coupling matrix'.format(rank))
        mc = nw.mode_coupling(namap_cut, namap_sig, bins, 
                    mcm_dir=opj(args.mcmdir, subdir), overwrite=False)

        print('[rank {:03d}]: calculating spectra'.format(rank))
        cb = nw.compute_spectra(namap_cut, namap_sig, mc=mc)
        cb_signal = nw.compute_spectra(namap_sig, namap_sig, mc=mc)
        cb_cut = nw.compute_spectra(namap_cut, namap_cut, mc=mc)

        print('[rank {:03d}]: saving spectra'.format(rank))
        np.save(opj(outdir, 'cb'), cb['TT'])
        np.save(opj(outdir, 'cb_signal'), cb_signal['TT'])
        np.save(opj(outdir, 'cb_cut'), cb_cut['TT'])
        np.save(opj(outdir, 'ell'), cb['ell'])

        # Repeat for source-free signal map.
        print('[rank {:03d}]: reading source-free map'.format(rank))
        
        if args.signal_map_srcfree is not None:
            signal_map = args.signal_map_srcfree
            if len(signal_map) == 1:
                signal_map = signal_map[0]
            signal = read_signal_map(signal_map, freq, ratio.shape, ratio.wcs)

            print('[rank {:03d}]: processing source-free map'.format(rank))
            if args.downgrade != 1:
                signal = enmap.downgrade(signal, args.downgrade)

            namap_sig = nw.namap_car(maps=(signal, None, None), masks=mask)

            print('[rank {:03d}]: calculating spectra source-free map'.format(rank))
            cb = nw.compute_spectra(namap_cut, namap_sig, mc=mc)
            cb_signal = nw.compute_spectra(namap_sig, namap_sig, mc=mc)

            print('[rank {:03d}]: saving spectra source-free map'.format(rank))
            np.save(opj(outdir, 'cb_srcfree'), cb['TT'])                        
            np.save(opj(outdir, 'cb_signal_srcfree'), cb_signal['TT'])

