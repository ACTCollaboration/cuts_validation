import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from mpi4py import MPI

from pixell import enmap, enplot
from enlib import config

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = config.ArgumentParser()
    parser.add_argument("sdir")
    parser.add_argument("-d", "--downgrade", type=float, default=4)
    parser.add_argument("-r", "--range", type=float, default=0.02)
    args = parser.parse_args()

    plot_opts = {'quantile' : 0.001, 'colorbar' : True, 'ticks' : 5,
                 'mask' : np.nan, 'autocrop' : True, 'color' : 'gray'}

    mapdirs = glob.glob(opj(args.sdir, 'pa*_f*_s*'))

    for mapdir in mapdirs[comm.Get_rank():len(mapdirs)+1:comm.Get_size()]:

        print('rank {:3d}: plotting {}'.format(
            comm.Get_rank(), os.path.split(mapdir)[1]))

        hitmap = enmap.read_map(opj(mapdir, 'hits.fits'))
        cutmap = enmap.read_map(opj(mapdir, 'cuts.fits'))        
        ratio = cutmap.copy()
        ratio[hitmap == 0] *= 0
        ratio[hitmap != 0] /= hitmap[hitmap != 0]

        hitmap = enmap.downgrade(hitmap, args.downgrade)
        cutmap = enmap.downgrade(cutmap, args.downgrade)
        ratio = enmap.downgrade(ratio, args.downgrade)

        ratio[hitmap == 0] = np.nan

        plot = enplot.plot(hitmap, **plot_opts)
        enplot.write(opj(mapdir, 'hits'), plot)

        plot = enplot.plot(cutmap, **plot_opts)
        enplot.write(opj(mapdir, 'cuts'), plot)

        plot = enplot.plot(ratio, min=0, max=args.range, **plot_opts)
        enplot.write(opj(mapdir, 'ratio'), plot)




    
