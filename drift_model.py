#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8



import numpy as np


def normalize(ux, uy, limit=None):
    angle = np.arctan2(uy, ux)
    uxn = np.cos(angle)
    uyn = np.sin(angle)

    if limit != None:
        norm = np.linalg.norm([ux,uy])
        new_norm = np.max([norm, limit])
        uxn *= new_norm
        uyn *= new_norm

    return [uxn, uyn]


def ripple(x,y, xoff, yoff, xscale, yscale):
    xi = (x*xscale) - xoff
    yi = (y*yscale) - yoff

    ux = np.cos(xi*0.1 + 100)
    uy = np.sin(np.sqrt(yi**2 + xi**2))

    # ux = np.cos(x+y)
    # uy = np.sin(np.sqrt(y**2 + x**2))
    # return [ux, uy]

    return [ux, uy]


def spiral(x, y, xoff, yoff, xscale, yscale):
    xi = x - xoff
    yi = y - yoff

    ux = yi-xi
    uy = -xi-yi

    return [ux*xscale, uy*yscale]



def mix_functions(x, y, funcs, xoffs, yoffs, xscales, yscales):
    ux = np.zeros_like(x)
    uy = np.zeros_like(y)

    for func, xoff, yoff, xscale, yscale in zip(funcs, xoffs, yoffs, xscales, yscales):

        uxi, uyi = func(x, y, xoff, yoff, xscale, yscale)
        uxi, uyi = normalize(uxi, uyi)

        xdiff = x - xoff
        ydiff = y - yoff
        dists = np.linalg.norm([xdiff, ydiff], axis=0)

        uxi /= dists
        uyi /= dists

        ux += uxi
        uy += uyi

    return [ux, uy]



class DriftModel:
    def __init__(self,
                 num_spirals,
                 num_ripples,
                 area_xsize,
                 area_ysize,
                 scale_size = 100):

        self.funcs = num_spirals*[spiral] + num_ripples*[ripple]
        self.area_xsize = area_xsize
        self.area_ysize = area_ysize

        num_elements = len(self.funcs)
        self.xcenters = []
        self.ycenters = []
        for i in range(num_elements):
            for trial in range(10):
                x = area_xsize*np.random.random()
                y = area_ysize*np.random.random()

                dists = []
                for xc,yc in zip(self.xcenters, self.ycenters):
                    xdiff = x-xc
                    ydiff = y-yc
                    dist = np.linalg.norm([xdiff, ydiff])
                    dists.append(dist)

                dists = np.array(dists)

                if i==0 or all(dists > min(area_xsize, area_ysize)/10):
                    self.xcenters.append(x)
                    self.ycenters.append(y)
                    break

        self.xscales = scale_size*2*(np.random.random(num_elements)-0.5)
        self.yscales = scale_size*2*(np.random.random(num_elements)-0.5)

    def sample(self,xs,ys):

        uxs, uys = mix_functions(xs,ys,
                                 self.funcs,
                                 self.xcenters,
                                 self.ycenters,
                                 self.xscales,
                                 self.yscales)

        uxs, uys = normalize(uxs, uys, limit=100)

        return uxs, uys


    def visualize(self, ax, meters_between_arrows):
        x_count = int(self.area_xsize / meters_between_arrows)
        y_count = int(self.area_ysize / meters_between_arrows)

        x_axis = np.linspace(0, self.area_xsize, x_count)
        y_axis = np.linspace(0, self.area_ysize, y_count)
        xs, ys = np.meshgrid(x_axis, y_axis)

        uxs, uys = self.sample(xs, ys)

        colors = np.arctan2(uys, uxs)
        ax.quiver(xs, ys, uxs, uys, colors, cmap='hsv')
        ax.scatter(self.xcenters, self.ycenters, c='k')
        for x,y, xsc, ysc in zip(self.xcenters, self.ycenters, self.xscales, self.yscales):
            e = patches.Ellipse((x,y), xsc*10, ysc*10, fill=True, alpha=0.2, color='b')
            ax.add_patch(e)






if __name__ == '__main__':

    import sys
    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams['pdf.fonttype'] = 42
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    try:
        seed = int(sys.argv[1])
    except:
        import time
        seed = int(time.time())
        print(f'Seed={seed}')

    np.random.seed(seed)



    drift_model = DriftModel(num_spirals = 4,
                             num_ripples = 0,
                             area_xsize = 2000,
                             area_ysize = 1000,
                             scale_size = 100)

    drift_model.visualize(ax, meters_between_arrows = 20)







