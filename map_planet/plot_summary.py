'''
Make summary plots for all TOD subfolders.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys
import re

from mpi4py import MPI

opj = os.path.join

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_dets(filename):

    dets = np.loadtxt(filename, dtype=str, usecols=0, skiprows=1)
    return dets

def read_noise(filename):

    var = np.loadtxt(filename, usecols=1, skiprows=1)
    return var

def read_rcal(filename):

    rcal = np.loadtxt(filename, usecols=2, skiprows=1)
    return rcal

def plot_hist(outfile, noise, xlabel=None):

    fig, ax = plt.subplots()
    ax.hist(noise, bins=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('no. detectors')
    ax.set_ylim(bottom=-2)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

def plot_scatter(outfile, x, y, xlabel=None, ylabel=None):

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    
if __name__ == '__main__':
    
    # Read commandline to get output directory.
    try:
        outdir = sys.argv[1]
    except IndexError as e:
        if rank == 0:
            print('Usage: python plot_det_var.py outdir')
        raise e

    # Find all subdirectories on root.
    if rank == 0:
        tod_dirs = glob.glob(opj(outdir, '*.*.ar*'))
    else:
        tod_dirs = None

    # Broadcast subdirectories.
    tod_dirs = comm.bcast(tod_dirs, root=0)
    tod_dirs_sub = tod_dirs[rank::size]

    for tod_dir in tod_dirs_sub:
        detfiles = glob.glob(opj(tod_dir, 'det_stats_f*.txt'))
        
        for detfile in detfiles:

            path, fname = os.path.split(detfile)
            freq_str = re.search('det_stats_f(.+?).txt', fname).group(1)

            dets = read_dets(detfile)
            var = read_noise(detfile)
            rcal = read_rcal(detfile)

            noise_std = np.sqrt(var)

            noisefile = opj(path, 'noise_hist_f{}'.format(freq_str))
            plot_hist(noisefile, np.log10(noise_std), xlabel='log stdev. noise [uK]')

            rcalfile = opj(path, 'rcal_hist_f{}'.format(freq_str))
            plot_hist(rcalfile, rcal, xlabel='rel. cal [pW/DAC]')

            scatterfile = opj(path, 'rcal_vs_noise_f{}'.format(freq_str))
            plot_scatter(scatterfile, np.log10(noise_std), 
                         np.log10(np.abs(rcal)), xlabel='log stdev. noise [uK]',
                         ylabel='log abs(rel. cal) [pW/DAC]')
