#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from mpi4py import MPI

from pixell import enmap, enplot, wcsutils
from enlib import config

comm = MPI.COMM_WORLD
opj = os.path.join

try:
    basestring
except NameError:
    basestring = str

def read_signal_map(signal_map, freq, shape=None, wcs=None):
    '''
    Return signal map for given frequency. Optionally, 
    crop to match shape of other (smaller) map.
    
    Parameters
    ----------
    signal_map : str or sequence of strings.
        Path to .fits file or list of paths. Filenames
        must contain unique "fXXX" identifier. If single
        path is given, must contain "{freq}" as placeholder.
    freq : str
        Frequency, e.g. "f150".
    shape : tuple, optional
        Desired shape.
    wcs : astropy.wcs.wcs.WCS object, optional
        WCS of desired cropped map.

    Return
    ------
    signal_map : enmap
        Signal map for given frequency.

    Raises
    ------
    ValueError
        If no suitable signal map is found.
    '''
    
    signal_file = None
    # Input is string or list of strings.
    if isinstance(signal_map, basestring):
        signal_file = signal_map.replace('{freq}', freq)
    else:
        for path in signal_map:
            _, filename = os.path.split(path)
            if freq in filename:
                signal_file = path
                break
    if signal_file is None:
        raise ValueError('No signal map found for freq: {}'.format(freq))

    if shape is not None and wcs is not None:
        shape_signal, wcs_signal = enmap.read_map_geometry(signal_file)
        sel = get_slice(shape[-2::], wcs, shape_signal, wcs_signal)
    else:
        sel = None

    signal = enmap.read_map(signal_file, sel=sel)

    return signal

def get_slice(shape_small, wcs_small, shape_large, wcs_large):
    '''
    Return slice of bigger map corresponding to smaller map.
    If bigger map includes I, Q, U, etc. dimensions, only the
    first one is selected.
    
    shape_small : tuple
        Shape of small map.
    wcs_small : astropy.wcs.wcs.WCS object, optional
        WCS of small map.
    shape_large : tuple
        Shape of large map.
    wcs_large : astropy.wcs.wcs.WCS object, optional
        WCS of large map.

    returns
    -------
    slice : slice object
        Slice into bigger map.

    Raises
    ------
    ValueError
        If WCSs of maps are incompatible.
        If small map is not 2d.
        If large map is smaller than small map.
    '''

    if not wcsutils.is_compatible(wcs_small, wcs_large):
        raise ValueError('Incompatible WCSs.')
    if len(shape_small) != 2:
        raise ValueError('Small map needs to be 2d.')
    if shape_large[-1] < shape_small[-1] or shape_large[-2] < shape_small[-2]:
        raise ValueError('Large map is smaller than small map.')

    if shape_small == shape_large:
        return Ellipsis
    elif shape_small == shape_large[-2::]:
        return (0,) * (len(shape_large) - 2) + np.s_[:,:]
    else:
        skybox = enmap.box(shape_small, wcs_small)
        pixbox = enmap.skybox2pixbox(shape_large, wcs_large, skybox, corner=True)
        pixbox = pixbox.astype(int)
        return (0,) * (len(shape_large) - 2) + np.s_[pixbox[0,0]:pixbox[1,0],
                                                    pixbox[0,1]:pixbox[1,1]]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("sdir", 
                        help='Source directory containing "pa*_f*_s*" directories.')
    parser.add_argument("-d", "--downgrade", type=float, default=4,
                        help='Downgrade maps by this factor before plotting.')
    parser.add_argument("-r", "--range", type=float, default=0.02,
                        help='Upper limit of colorbar of ratio plot.')
    parser.add_argument("-s", "--signal-map", nargs='+',
                        help='Path to .fits file to be overplotted with cut ratio contours. '
                        'Can be list of paths for different freqs or single path with '
                        '"{freq}" in filename.') 
    args = parser.parse_args()

    plot_opts = {'quantile' : 0.001, 'colorbar' : True, 'ticks' : 5,
                 'mask' : np.nan, 'autocrop' : False, 'color' : 'gray'}

    mapdirs = glob.glob(opj(args.sdir, 'pa*_f*_s*'))

    for mapdir in mapdirs[comm.Get_rank():len(mapdirs)+1:comm.Get_size()]:

        print('rank {:3d}: plotting {}'.format(
            comm.Get_rank(), os.path.split(mapdir)[1]))

        hitmap = enmap.read_map(opj(mapdir, 'hits.fits'))
        cutmap = enmap.read_map(opj(mapdir, 'cuts.fits'))
        ratio = cutmap.copy()
        ratio[hitmap == 0] *= 0
        ratio[hitmap != 0] /= hitmap[hitmap != 0]

        if args.signal_map is not None:

            freq = [x for x in mapdir.split('_') if x.startswith('f')][0]
            signal_map = args.signal_map
            if len(signal_map) == 1:
                signal_map = signal_map[0]
            signal = read_signal_map(signal_map, freq, ratio.shape, ratio.wcs)
            signal = enmap.downgrade(signal, args.downgrade)            
        
        hitmap = enmap.downgrade(hitmap, args.downgrade)
        cutmap = enmap.downgrade(cutmap, args.downgrade)
        ratio = enmap.downgrade(ratio, args.downgrade)

        ratio[hitmap == 0] = np.nan

        plot = enplot.plot(hitmap, **plot_opts)
        enplot.write(opj(mapdir, 'hits'), plot)

        plot = enplot.plot(cutmap, **plot_opts)
        enplot.write(opj(mapdir, 'cuts'), plot)

        plot = enplot.plot(ratio, min=0,  max=args.range, **plot_opts)
        enplot.write(opj(mapdir, 'ratio'), plot)

        if args.signal_map is not None:

            ratio[hitmap == 0] = 0.
            ratio_sm = enmap.downgrade(ratio, 32 / float(args.downgrade))
            ratio_sm = enmap.project(ratio_sm, ratio.shape, ratio.wcs, order=1)
        
            contours = '0.003, 0.01'
            contour_layer = enplot.plot(ratio_sm, contours=contours, no_image=True, 
                                        contour_width=2, contour_color='hotcold',
                                        min=0, max=args.range, **plot_opts)
            enplot.write(opj(mapdir, 'ratio_contour'), contour_layer)

            signal_layer = enplot.plot(signal, layers=True, min=-300, max=3000,
                                       reverse_color=True, **plot_opts)

            enplot.write(opj(mapdir, 'signal_ratio_contour.png'),
                         enplot.merge_plots(signal_layer+contour_layer))
