#!/usr/bin/env python  
'''
Plot the cross correlation between a sky map and the cut (ratio) map. 
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import argparse
import glob
import numpy as np
from scipy.interpolate import CubicSpline
from mpi4py import MPI

from pixell import utils

opj = os.path.join
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def subplots(**kwargs):
    '''
    Wrapper around plt.subplots that defaults to inner tick labels.

    Parameters
    ----------
    kwargs : dict, optional
        Keyword arguments to plt.subplots.

    Returns
    -------
    fig, axs
    '''
    
    fig, axs = plt.subplots(**kwargs)
    for ax in np.ravel(axs):
        ax.tick_params(direction='in', right=True, top=True)
    return fig, axs
    
def get_subdirs(basedir, pattern='pa*_f*_s*'):
    '''
    Return sub directories with names that match given pattern.

    Parameters
    ----------
    basedir : str
        Look for directories in this directory 
    pattern : str
        Pattern to be matched by filenames, can contain wildcards.
    
    Returns
    -------
    subdirs : list
        Absolute paths to sub directories.
    '''

    files = glob.glob(opj(basedir, pattern))
    subdirs = [os.path.split(f)[1] for f in files]
    
    return subdirs

def cl2ct(cl, bins, theta):
    '''
    Transform angular power spectrum C_ell to correlation function C(theta).

    C(theta) = sum_l=2^lmax (2l+1)/4pi * Cl * Pl(cos theta).

    Parameters
    ----------
    cl : array
        Angular power spectrum.
    bins : array
        Multipole bins corresponding to power spectrum
    theta : theta
        Output opening angles.
    
    Returns
    -------
    ct : array
        Angular correlation function.
    '''

    bins = bins.astype(int)

    # We want lmin=2 or higher.
    try:
        idx_min = np.where(bins == 2)[0][0]
    except IndexError:
        idx_min = 0
    bins = bins[idx_min:]
    cl = cl[idx_min:]

    if bins.size != bins[-1] - 1:
        # Requires interpolation.
        cs = CubicSpline(bins, cl)
        ells_full = np.arange(bins[0], bins[-1] + 1)
        cl = cs(ells_full)
        ells = ells_full
        
    # Now we add l=0 and l=1, or the ells before the first bin back in.
    cl_full = np.zeros(ells[-1]+1)
    cl_full[ells[0]:] = cl
    ells_full = np.arange(ells[-1] + 1)

    prefactor = (2 * ells_full + 1) / 4. / np.pi

    return np.polynomial.legendre.legval(np.cos(theta), prefactor * cl_full)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("sdir", help='Spectra directory')
    parser.add_argument("odir", help='Output directory')
    parser.add_argument("--min-bin", type=int, default=0,
            help='Specify the index of first bin to be plotted')
    args = parser.parse_args()

    if rank == 0:
        utils.mkdir(args.odir)

    subdirs = get_subdirs(args.sdir)

    for subdir in subdirs[rank::comm.Get_size()]:
        
        print('[rank {:03d}]: plotting spectra for {}'.format(rank, subdir))
        outdir = opj(args.odir, subdir)
        utils.mkdir(outdir)

        ells = np.load(opj(args.sdir, subdir, 'ell.npy'))
        theta = np.radians(np.logspace(-3, np.log10(179), num=4000))

        # Cross spectra.
        cl = np.load(opj(args.sdir, subdir, 'cb.npy'))

        # Auto spectra.
        cl_cut = np.load(
            opj(args.sdir, subdir, 'cb_cut.npy'))
        cl_signal = np.load(
            opj(args.sdir, subdir, 'cb_signal.npy'))

        try:
            cl_srcfree = np.load(
                opj(args.sdir, subdir, 'cb_srcfree.npy'))
            cl_signal_srcfree = np.load(
                opj(args.sdir, subdir, 'cb_signal_srcfree.npy'))
            srcfree = True
        except EnvironmentError:
            srcfree = False

        if args.min_bin != 0:
            ells = ells[args.min_bin:]
            cl = cl[args.min_bin:]
            cl_cut = cl_cut[args.min_bin:]
            cl_signal = cl_signal[args.min_bin:]
            if srcfree:
                cl_srcfree = cl_srcfree[args.min_bin:]
                cl_signal_srcfree = cl_signal_srcfree[args.min_bin:]

        rho = cl / np.sqrt(np.abs(cl_cut) * np.abs(cl_signal))
        if srcfree:
            rho_srcfree = cl_srcfree / np.sqrt(np.abs(cl_cut) * np.abs(cl_signal_srcfree))

        ct = cl2ct(cl, ells, theta)
        ct_cut = cl2ct(cl_cut, ells, theta)
        ct_signal = cl2ct(cl_signal, ells, theta)
        if srcfree:
            ct_srcfree = cl2ct(cl_srcfree, ells, theta)
            ct_signal_srcfree = cl2ct(cl_signal_srcfree, ells, theta)

        rho_theta = ct / np.sqrt(np.abs(ct_cut[0]) * np.abs(ct_signal[0]))
        rho_theta_srcfree = ct_srcfree / np.sqrt(np.abs(ct_cut[0]) * np.abs(ct_signal_srcfree[0]))

        dells = ells * (ells + 1) / 2. / np.pi

        # Multipole plot.
        fig, axs = subplots(nrows=4, sharex=True)

        # Auto spectra.
        axs[0].plot(ells, dells * cl_cut, label='cuts / hits', color='black')
        axs[1].plot(ells, dells * cl_signal, label='signal')
        if srcfree:
            axs[1].plot(ells, dells * cl_signal_srcfree, 
                        label='signal_srcfree', ls='--')
        axs[0].set_ylabel(r'$D_{\ell}$')
        axs[1].set_ylabel(r'$D_{\ell}$ [$\mathrm{\mu K_{CMB}}^2$]')

        # Cross spectra.
        axs[2].plot(ells, dells * cl, label='signal x (cuts / hits)')
        if srcfree:
            axs[2].plot(ells, dells * cl_srcfree,
                        label='signal_srcfree x (cuts / hits)', ls='--')

        axs[2].set_ylabel(r'$D_{\ell}$ [$\mathrm{\mu K_{CMB}}$]')

        # Cross correlation coefficient.
        axs[3].plot(ells, rho,
                    label=r'$\rho_{\ell}^{\mathrm{(cuts/hits)}, \mathrm{signal}}$')
        if srcfree:
            axs[3].plot(ells, rho_srcfree, ls='--',
                        label=r'$\rho_{\ell}^{\mathrm{(cuts/hits)}, \mathrm{signal\_srcfree}}$')

        axs[0].set_title(subdir)
        axs[3].set_ylabel(r'xcorr. coeff.')
        axs[3].set_xlabel(r'Multipole $\ell$')
        for ax in axs:
            ax.legend(frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fig.savefig(opj(outdir, 'cl_cuts.png'), dpi=250)
        plt.close(fig)


        # Real space plot.
        theta_deg = np.degrees(theta)
        fig, axs = subplots(nrows=4, sharex=True)

        # Auto spectra.
        axs[0].plot(theta_deg, ct_cut, label='cuts / hits', color='black')
        axs[1].plot(theta_deg, ct_signal, label='signal')
        if srcfree:
            axs[1].plot(theta_deg, ct_signal_srcfree, 
                        label='signal_srcfree', ls='--')
        axs[0].set_ylabel(r'$C(\theta)$')
        axs[1].set_ylabel(r'$C(\theta)$ [$\mathrm{\mu K_{CMB}}^2$]')

        # Cross spectra.
        axs[2].plot(theta_deg, ct, label='signal x (cuts / hits)')
        if srcfree:
            axs[2].plot(theta_deg, ct_srcfree,
                        label='signal_srcfree x (cuts / hits)', ls='--')

        axs[2].set_ylabel(r'$C(\theta)$ [$\mathrm{\mu K_{CMB}}$]')

        # Cross correlation coefficient.
        axs[3].plot(theta_deg, rho_theta,
                    label=r'$\rho^{\mathrm{(cuts/hits)}, \mathrm{signal}}(\theta)$')
        if srcfree:
            axs[3].plot(theta_deg, rho_theta_srcfree, ls='--',
                        label=r'$\rho^{\mathrm{(cuts/hits)}, \mathrm{signal\_srcfree}}(\theta)$')

        axs[3].set_xscale('log')
        axs[0].set_title(subdir)
        axs[3].set_ylabel(r'xcorr. coeff.')
        axs[3].set_xlabel(r'Opening angle $\theta$ [$\mathrm{deg}$]')
        for ax in axs:
            ax.legend(frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fig.savefig(opj(outdir, 'ct_cuts.png'), dpi=250)
        plt.close(fig)
