#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scikit-learn
#import seaborn as sns
from scipy.optimize import leastsq
from scipy import interpolate

_order = {1:3, 2:6, 3:10, 4:15}
_num2order = {3:1, 6:2, 10:3, 15:4}
_coef2order = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:3, 10:4,
               11:4, 12:4, 13:4, 14:4}

class Mapwin(object):
    """Base class for Regi, GMC, and Map."""
    def __init__(self):
        self.type = 'mapwin'

    def __add__(self, other):
        if self.type == 'regi' and other.type == 'regi':
            if (self.xgrid == other.xgrid) and\
                    (self.ygrid == other.ygrid) and\
                    (self.xpitch == other.xpitch) and\
                    (self.ypitch == other.ypitch):
                tmp = copy.deepcopy(self)
                tmp.data.xdev = self.data.xdev + other.data.xdev
                tmp.data.ydev = self.data.ydev + other.data.ydev
                tmp.xmean = tmp.data.xdev.mean()
                tmp.ymean = tmp.data.ydev.mean()
                #print("mean %f:%f" %(tmp.xmean, tmp.ymean))
                tmp.name = ' + '.join([self.name,other.name])
                return tmp
            else:
                print('grid and/or pitch doesn\'t match')
                return 1

        elif self.type == 'gmc' and other.type == 'gmc':
            tmp = copy.deepcopy(self)
            tmp.a = self.a + other.a
            tmp.b = self.b + other.b
            return tmp

        elif self.type == 'map' and other.type == 'map':
            pass
        elif (self.type == 'regi' and other.type == 'gmc') and\
                (self.type == 'gmc' and other.type == 'regi'):
            pass
        elif (self.type == 'gmc' and other.type == 'map') and\
                (self.type == 'map' and other.type == 'gmc'):
            pass
        elif (self.type == 'gmc' and other.type == 'map') and\
                (self.type == 'map' and other.type == 'gmc'):
            pass


    def __sub__(self, other):
        if self.type == 'regi' and other.type == 'regi':
            if (self.xgrid == other.xgrid) and\
                    (self.ygrid == other.ygrid) and\
                    (self.xpitch == other.xpitch) and\
                    (self.ypitch == other.ypitch):
                tmp = copy.deepcopy(self)
                tmp.data.xdev = self.data.xdev - other.data.xdev
                tmp.data.ydev = self.data.ydev - other.data.ydev
                tmp.xmean = tmp.data.xdev.mean()
                tmp.ymean = tmp.data.ydev.mean()
                #print(tmp.xmean, tmp.ymean)
                tmp.name = ' - '.join([self.name,other.name])
                return tmp
            else:
                print('grid and/or pitch doesn\'t match')
                return 1

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            tmp = copy.deepcopy(self)
            tmp.data.xdev *= float(other)
            tmp.data.ydev *= float(other)
            tmp.xmean = tmp.data.xdev.mean()
            tmp.ymean = tmp.data.ydev.mean()
            return tmp

    def __div__(self, other):
        if self.type == 'regi':
            if type(other) == int or type(other) == float:
                tmp = copy.deepcopy(self)
                tmp.data.xdev /= float(other)
                tmp.data.ydev /= float(other)
                tmp.xmean = tmp.data.xdev.mean()
                tmp.ymean = tmp.data.ydev.mean()
                return tmp

    def __truediv__(self, other):
        if self.type == 'regi':
            if type(other) == int or type(other) == float:
                tmp = copy.deepcopy(self)
                tmp.data.xdev /= float(other)
                tmp.data.ydev /= float(other)
                tmp.xmean = tmp.data.xdev.mean()
                tmp.ymean = tmp.data.ydev.mean()
                return tmp

    def make_grid(self, gridnum, pitch):
        if gridnum % 2 == 0:
            tmpgrid = np.array([[(-1.0*pitch)*gridnum/2.0+pitch/2.0+pitch*x,
                pitch*(gridnum/2.0)-pitch/2.0-pitch*y]
                for y in range(0, gridnum) for x in range(0, gridnum)])
        else:
            tmpgrid = np.array([[(-1.0*pitch)*((gridnum-1.0)/2.0)+pitch*x,
                pitch*((gridnum-1.0)/2.0)-pitch*y]
                for y in range(0, gridnum) for x in range(0, gridnum)])
        return tmpgrid

    def make_interpolation(self, kind='linear'):
        """ To match with IPRO data format, x grid starts from minus, and Y
        grid starts from plus.
        """
        tempx = self.data.x.copy()
        tempy = self.data.y.copy()
        # Acending: from minus to plus
        old_x_grid = tempx.sort_values(ascending=True).unique()
        # Decending: from plus to minus
        old_y_grid = tempy.sort_values(ascending=False).unique()

        xx, yy = np.meshgrid(old_x_grid, old_y_grid)
        zzx = self.data.xdev.reshape(self.xgrid, self.ygrid)
        zzy = self.data.ydev.reshape(self.xgrid, self.ygrid)

        f = interpolate.interp2d(old_x_grid, old_y_grid, zzx, kind=kind)
        g = interpolate.interp2d(old_x_grid, old_y_grid, zzy, kind=kind)
        return f, g


class Gmc(Mapwin):
    def __init__(self):
        self.type = 'gmc'
        self.name = 'gmc param'
        # prepare coef for 4th order polynomial equation
        self.a = np.zeros(_order[4])
        self.b = np.zeros(_order[4])

    def load_gmc_data(self, mapwin_gmc_file):
        """ load mapwin-formated gmc file"""
        #df = pd.read_table(mapwin_gmc_file, delimitor='=', comment='//')
        with open(mapwin_gmc_file, 'r') as f:
            for line in f:
                #index = '{0:s}'.format(line.split('=')[0])
                #coef = '{0:s}'.format(line.split('=')[1])
                #print index, coef
                if line[0:2] == '//':
                    pass
                elif line[0:1] == 'A':
                    splitline = line.split('=')
                    index = splitline[0]
                    coef = splitline[1]
                    self.a[int(index[1:])] = float(coef)
                elif line[0:1] == 'B':
                    splitline = line.split('=')
                    index = splitline[0]
                    coef = splitline[1]
                    self.b[int(index[1:])] = float(coef)

    def to_gmcfile(self, filename='gmcparam.txt', order=3):
        with open(filename, 'w') as file:
            text = '// GMC Parameter of {name:s}\n'.format(name = self.name)
            file.write(text)
            if order == 3:
                for i in range(0, len(self.a)):
                    text = 'A{index:d}={num:e}\n'.format(index=i, num=self.a[i])
                    file.write(text)
                    file.write('\n')
                for i in range(0, len(self.b)):
                    text = 'A{index:d}={num:e}\n'.format(index=i, num=self.b[i])
                    file.write(text)

    def gmc2regi(self, grid='', pitch='', unit='nm'):
        """
        This method converts gmc coefficient into registration data, which is
        like mapwin graph.
        """
        tmpgrid = self.make_grid(gridnum=grid, pitch=pitch)

        if unit == 'nm':
            px = self.a*1e-3  #convert um to nm
            py = self.b*1e-3  #convert um to nm
            mpl = 1e3 # multiplicator
        elif unit == 'um':
            px = self.a
            py = self.b
            mpl = 1 # multiplicator
        else:
            print("Unit must be either of 'nm' or 'um'")
            sys.exit(1)

        print("Unit: %s" %unit)
        print(px, py)

        x = tmpgrid[:,0]
        y = tmpgrid[:,1]
        length = len(tmpgrid)

        order = _num2order[len(self.a)]
        print('order: ', order)
        exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]
        for i, exp in enumerate(exps):
            xdev = px[i] * x[i] ** exp[0] * y[i] ** exp[1] * mpl ** (exp[0] + exp[1])
            ydev = py[i] * x[i] ** exp[0] * y[i] ** exp[1] * mpl ** (exp[0] + exp[1])

        _regi = Regi()
        _regi.name = 'gmc2regi(order:{order:d})'.format(order=order)
        _regi.xgrid = grid
        _regi.ygrid = grid
        _regi.xpitch = pitch
        _regi.ypitch = pitch
        _regi.data = pd.DataFrame({'x':x, 'y':y, 'xdev':xdev, 'ydev':ydev})
        _regi.xmean = _regi.data.xdev.mean()
        _regi.ymean = _regi.data.ydev.mean()
        return _regi


class Map(Mapwin):
    def __init__(self, grid='', pitch=''):
        self.type = 'map'
        self.xgrid = grid
        self.ygrid = grid
        self.name = 'map {0:d}x{1:d}'.format(int(self.xgrid), int(self.ygrid))
        self.xpitch = pitch
        self.ypitch = pitch

    def load_map_data(self, mapwin_map_file):
        data = np.loadtxt(mapwin_map_file, skiprows=9)
        with open(mapwin_map_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 2:
                    self.xpitch = float(line.split('=')[1])/1e3
                    # pitch is nm unit in map file so divide it by 1e3 to
                    # convert it to um unit.
                elif i == 3:
                    self.ypitch = float(line.split('=')[1])/1e3
                    # pitch is nm unit in map file so divide it by 1e3 to
                    # convert it to um unit.
                elif i == 4:
                    self.xgrid = int(line.split('=')[1])
                elif i == 5:
                    self.ygrid = int(line.split('=')[1])

        self.name = mapwin_map_file

        grid = self.make_grid(self.xgrid, self.xpitch)
        data = np.reshape(data, (len(data), 1))

        xdevini = 0
        xdevend = xdevini + self.xgrid * self.ygrid
        ydevini = xdevend
        ydevend = ydevini + self.xgrid * self.ygrid

        xdev = data[xdevini:xdevend]/1e3
        ydev = data[ydevini:ydevend]/1e3
        # deviation is nm unit in map file so to be same as mapwin format file,
        # divide it with 1e3 to make it um unit.
        self.data = pd.DataFrame(np.concatenate((grid, xdev, ydev), axis=1),
                                 columns=['x','y','xdev','ydev'])

    def map2regi(self):
        _regi = Regi()
        _regi.name = 'map2regi {xgrid:d}x{ygrid:d}'\
            .format(xgrid=self.xgrid, ygrid=self.ygrid)
        _regi.xgrid = self.xgrid
        _regi.ygrid = self.ygrid
        _regi.xpitch = self.xpitch
        _regi.ypitch = self.ypitch
        _regi.data = self.data
        _regi.xmean = _regi.data.xdev.mean()
        _regi.ymean = _regi.data.ydev.mean()
        return _regi

    def to_mapfile(self, filename='mapcorr.txt'):
        with open(filename, 'w') as file:
            text = '// Map Correction File from {mapfile:s}\n'\
                .format(mapfile=self.name)
            file.write(text)
            file.write('HEAD\n')
            file.write('PITCH_X={xpitch:0.0f}\n'\
                       .format(xpitch=self.xpitch*1e3))
            file.write('PITCH_Y={ypitch:0.0f}\n'\
                       .format(ypitch=self.ypitch*1e3))
            file.write('GRID_X={xgrid:d}\n'.format(xgrid=self.xgrid))
            file.write('GRID_Y={ygrid:d}\n'.format(ygrid=self.ygrid))
            file.write('ORIGIN_X={xorigin:0.0f}\n'\
                    .format(xorigin=self.data.x[0]*1e3))
            file.write('ORIGIN_Y={yorigin:0.0f}\n'\
                    .format(yorigin=self.data.y[0]*1e3))
            file.write('MAP\n')
            for i in range(0, len(self.data.x)):
                    file.write('{xmap:0.2f}\n'\
                               .format(xmap=self.data.xdev[i]*1e3))
            for i in range(0, len(self.data.y)):
                    file.write('{ymap:0.2f}\n'\
                               .format(ymap=self.data.ydev[i]*1e3))

    def convert_grid(self, grid='', pitch='', kind='cubic'):
        """
        This method makes new grid data with new grid number and pitch. New grid
        data must be smaller than or equal to the original grid data.
        """
        old_x_max = max(self.data.x)
        old_x_min = min(self.data.x)
        old_y_max = max(self.data.y)
        old_y_min = min(self.data.y)

        tmpgrid = self.make_grid(gridnum=grid, pitch=pitch)

        new_x_grid = tmpgrid[:,0]
        new_y_grid = tmpgrid[:,1]
        new_x_data = np.zeros(len(new_x_grid))
        new_y_data = np.zeros(len(new_y_grid))
        new_x_max = max(new_x_grid)
        new_x_min = min(new_x_grid)
        new_y_max = max(new_y_grid)
        new_y_min = min(new_y_grid)
        
        xmin = old_x_min <= new_x_min
        xmax = new_x_max <= old_x_max
        ymin = old_y_min <= new_y_min
        ymax = new_y_max <= old_y_max

        if xmin & xmax & ymin & ymax:
            print('The new grid area is within the old grid area')
            _newmap = Map(grid=grid, pitch=pitch)
            print(_newmap.name)
            _newmap.data = pd.DataFrame({'x':new_x_grid, 'y':new_y_grid,
                'xdev':new_x_data, 'ydev':new_y_data})
            total = grid**2

            index1 = np.arange(0, total, grid, dtype=int)
            index2 = np.arange(grid, total+0.1, grid, dtype=int)

            # To match with IPRO data format, x grid starts from minus, and Y
            # grid starts from plus.
            tempx = self.data.x.copy()
            tempy = self.data.y.copy()
            # Acending: from minus to plus 
            old_x_grid = tempx.sort_values(ascending=True).unique()
            # Decending: from plus to minus
            old_y_grid = tempy.sort_values(ascending=False).unique()

            xx, yy = np.meshgrid(old_x_grid, old_y_grid)
            zzx = self.data.xdev.reshape(self.xgrid, self.ygrid)
            zzy = self.data.ydev.reshape(self.xgrid, self.ygrid)

            f = interpolate.interp2d(old_x_grid, old_y_grid, zzx, kind=kind)
            g = interpolate.interp2d(old_x_grid, old_y_grid, zzy, kind=kind)

            xdev_temp = np.empty(0)
            ydev_temp = np.empty(0)

            for i, j in zip(index1, index2):
                x = _newmap.data[i:j].x.values
                y = _newmap.data[i:j].y.values
                xdev = f(x, y[0])
                ydev = g(x, y[0])
                xdev_temp = np.append(xdev_temp, xdev)
                ydev_temp = np.append(ydev_temp, ydev)
            _newmap.data.xdev = xdev_temp
            _newmap.data.ydev = ydev_temp

            return _newmap
        else:
            print('The new grid area is larger than that of old grid')
            return self


class Regi(Mapwin):
    def __init__(self, file = ''):
        self.type = 'regi'
        self.xgrid = 0
        self.ygrid = 0
        self.xpitch = 0
        self.ypitch = 0

    def load_data(self, mapwin_file):
        """ load mapwin-formated data and calcurate grid coordinates"""
        data = pd.read_csv(mapwin_file, skiprows=7, names=('data',))
        #data = np.loadtxt(mapwin_file, skiprows=7)

        with open(mapwin_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 1:
                    name  = line.strip()
                elif i == 2:
                    xgrid = int(line.strip())
                elif i == 3:
                    ygrid = int(line.strip())
                elif i == 4:
                    xpitch = float(line.strip())
                elif i == 5:
                    ypitch = float(line.strip())
                elif i == 6:
                    break

        grid = self.make_grid(xgrid, xpitch)
        print(np.shape(grid))
    
        # with this data manipulation, "data" has become (x, 1) which was
        # originally (x, )
        length = len(data)
        data = np.reshape(data, (length, 1))

        # making indexes for X and Y
        xdevini = 0
        xdevend = xdevini + xgrid * ygrid
        ydevini = xdevend
        ydevend = ydevini + xgrid * ygrid

        # devide data into both xdev and ydev
        xdev = data[xdevini:xdevend]
        ydev = data[ydevini:ydevend]
        
        self.data = pd.DataFrame(np.concatenate((grid, xdev, ydev), axis = 1),
                columns = ['x', 'y', 'xdev', 'ydev'])
        self.name = name
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.xpitch = xpitch
        self.ypitch = ypitch

        self.xmean = self.data.xdev.mean()
        self.ymean = self.data.ydev.mean()
        print('Successfully loaded: {0:s}.'.format(self.name))

    def shift(self):
        tmp = copy.deepcopy(self)
        tmp.data.xdev = self.data.xdev - self.xmean
        tmp.data.ydev = self.data.ydev - self.ymean
        tmp.xmean = tmp.data.xdev.mean()
        tmp.ymean = tmp.data.ydev.mean()
        tmp.name = 'shifted ' + self.name
        return tmp

    def plot(self, scale=5, normalized=False):
        """ initial settings
            grid scale  =  5nm
            Normalized grid = False
        """
        if self.xgrid != self.ygrid:
            print("Grid number of X and Y must be same")
            return 1

        xgrid = self.data.x.unique()
        ygrid = self.data.y.unique()
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ### ax.set_aspect('equal')
        ### plt.axes().set_aspect('equal')
        ax.plot(self.data.x, self.data.y, '+', color='gray')
        boolxy = np.logical_or(np.isnan(self.data.xdev),
                               np.isnan(self.data.ydev))
        ax.plot(self.data.x[boolxy],
                self.data.y[boolxy],
                '+', color='red')
        print("scale %f" % scale)
        for x in xgrid:
            ax.plot(((self.data[self.data.x == x].xdev)*1000)/scale
                    * self.xpitch + self.data[self.data.x == x].x,
                    ((self.data[self.data.x == x].ydev)*1000)/scale
                    * self.ypitch + self.data[self.data.x == x].y,
                    'b-', alpha = 0.8)
        for y in ygrid:
            ax.plot(((self.data[self.data.y == y].xdev)*1000)/scale
                    * self.xpitch + self.data[self.data.y == y].x,
                    ((self.data[self.data.y == y].ydev)*1000)/scale
                    * self.ypitch + self.data[self.data.y == y].y,
                    'b-', alpha = 0.8)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, 
            box.width, box.height*0.9])
        legend_list = ['grid 1div: {0:.0f}[nm]'.format(scale)]
        legend_list.append('missing')
        legend_list.append(self.name)
        ax.legend(legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fontsize=9, fancybox=True, shadow=True)
        #ax.legend(legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.1))
        #ax.legend(legend_list, loc = 'lower center',
        #          bbox_to_anchor = (0.5, -0.2), 
        #          borderaxespad=0, fontsize=9)
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        plt.grid(True)
        plt.show(block = False)

    def colorplot(self, xmin=-5, xmax=5, ymin=-5, ymax=5):
        if self.xgrid != self.ygrid:
            print("Grid number of X and Y must be same")
            return 1

        xgrid = self.data.x.unique()
        ygrid = self.data.y.unique()
        xgrid_max = np.max(xgrid)
        xgrid_min = np.min(xgrid)
        ygrid_max = np.max(ygrid)
        ygrid_min = np.min(ygrid)

        #X error plot
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121)
        cax1 = ax1.imshow(np.reshape(self.data.xdev*1e3,
                                     (self.xgrid, self.ygrid)),
                          extent=(xgrid_min, xgrid_max, ygrid_min, ygrid_max),
                          interpolation='none', cmap='bwr',
                          vmin=xmin, vmax=xmax)
        ax1.set_title('X error [nm]')
        ax1.grid()
        fig.colorbar(cax1)

        #Y error plot
        ax2 = fig.add_subplot(122)
        cax2 = ax2.imshow(np.reshape(self.data.ydev*1e3,
                                     (self.xgrid, self.ygrid)),
                          extent=(xgrid_min, xgrid_max, ygrid_min, ygrid_max),
                          interpolation='none', cmap='bwr',
                          vmin=ymin, vmax=ymax)
        ax2.set_title('Y error [nm]')
        ax2.grid()
        fig.colorbar(cax2)
        plt.show(block=False)

    def report(self):
        dropna=self.data.dropna()
        xpitch = np.polyfit(dropna.x.values, dropna.xdev.values, 1)[0]
        xrotation = np.polyfit(dropna.y.values, dropna.xdev.values, 1)[0]
        ypitch = np.polyfit(dropna.y.values, dropna.ydev.values, 1)[0]
        yrotation = np.polyfit(dropna.x.values, dropna.ydev.values, 1)[0] 

        print("X mean[nm]:\t\t{0:.2f}".format(self.xmean*1000))
        print("X 3sigma[nm]:\t\t{0:.2f}".format(self.data.xdev.std(ddof=0)*3*1000))
        print("X max[nm]:\t\t{0:.2f}".format(self.data.xdev.max()*1000))
        print("X min[nm]:\t\t{0:.2f}".format(self.data.xdev.min()*1000))
        print("")
        print("Y mean[nm]:\t\t{0:.2f}".format(self.ymean*1000))
        print("Y 3sigma[nm]:\t\t{0:.2f}".format(self.data.ydev.std(ddof=0)*3*1000))
        print("Y max[nm]:\t\t{0:.2f}".format(self.data.ydev.max()*1000))
        print("Y min[nm]:\t\t{0:.2f}".format(self.data.ydev.min()*1000))
        print("")
        print("X pitch[ppm]:\t\t{0:5.4f}".format(xpitch*1e6))
        print("Y pitch[ppm]:\t\t{0:5.4f}".format(ypitch*1e6))
        print("X rotation[10-5 rad]:\t{0:5.4f}".format(xrotation*1e5))
        print("Y rotation[10-5 rad]:\t{0:5.4f}".format(yrotation*1e5))

    def fitfunc(self, p, exps, mpl, x, y):
        total = 0
        for i, exp in enumerate(exps):
            total += p[i] * x**exp[0] * y**exp[1] * mpl**(exp[0] + exp[1])
        return total

    def report_gmc(self, order=3, filename='gmcparam.txt'):
        exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]
        p0 = [0] * len(exps)
        errfunc = lambda p, exps, mpl, x, y, z: self.fitfunc(p, exps, mpl, x, y) - z
        dropna = self.data.dropna()
        resx = leastsq(errfunc, p0[:], args = (exps, 1e3, dropna.x, dropna.y, dropna.xdev))
        resy = leastsq(errfunc, p0[:], args = (exps, 1e3, dropna.x, dropna.y, dropna.ydev))
        _gmc = Gmc()
        #_gmc = Gmc(order=order)
        #for i in range(0, _order[order]):
            #_gmc.a[i] = resx[0][i] * -1
            #_gmc.b[i] = resy[0][i] * -1

        """"to match with mapwin coefficient, multiplying 1e3 to derived
        coefficient is needed. I'm not sure the exact reason why this 1e3
        multiplication is done here.
        """
        _gmc.a = resx[0]*-1*1e3
        _gmc.b = resy[0]*-1*1e3
        #_gmc.a = resx[0]*-1 #original
        #_gmc.b = resy[0]*-1 #original

        with open(filename, 'w') as file:
            #print('// GMC Parameter of {name:s}'.format(name = self.name))
            text = '// GMC Parameter of {name:s}\n'.format(name = self.name)
            file.write(text)
            for i in range(0, len(_gmc.a)):
                #print('A{index:d}={num:e}'.format(index=i, num=_gmc.a[i]))
                text = 'A{index:d}={num:e}\n'.format(index=i, num=_gmc.a[i])
                file.write(text)
            #print('')
            file.write('\n')
            for i in range(0, len(_gmc.b)):
                #print('A{index:d}={num:e}'.format(index=i, num=_gmc.b[i]))
                text = 'B{index:d}={num:e}\n'.format(index=i, num=_gmc.b[i])
                file.write(text)
        return _gmc

    def fit(self, order=3, filename='fit.txt'):
        exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]
        p0 = [0] * len(exps)
        errfunc = lambda p, exps, mpl, x, y, z: fitfunc(p, exps, mpl, x, y) - z
        dropna = self.data.dropna()
        resx = leastsq(errfunc, p0[:], args = (exps, 1, dropna.x, dropna.y, dropna.xdev))
        resy = leastsq(errfunc, p0[:], args = (exps, 1, dropna.x, dropna.y, dropna.ydev))

        print(resx[0])
        print(resy[0])
        with open(filename, 'w') as f:
            f.write("x,y\n")
            for x,y in zip(resx[0], resy[0]):
                f.write("%e,%e\n" %(x, y))

        _gmc = Gmc()
        _gmc.a = resx[0]
        _gmc.b = resy[0]
        return _gmc


    def report_map(self, filename='mapcorr.txt'):
        x = self.data.x
        y = self.data.y
        xdev = -1 * self.data.xdev
        ydev = -1 * self.data.ydev
        _map = Map(grid=self.xgrid, pitch=self.xpitch)
        _map.name = self.name
        _map.xgrid = self.xgrid
        _map.ygrid = self.ygrid
        _map.xpitch = self.xpitch
        _map.ypitch = self.ypitch
        #_map.data = self.data
        _map.data = pd.DataFrame({'x':x, 'y':y, 'xdev':xdev, 'ydev':ydev})

        with open(filename, 'w') as file:
            text = '// Map Correction File from {mapfile:s}\n'\
                .format(mapfile=_map.name)
            file.write(text)
            file.write('HEAD\n')
            file.write('PITCH_X={xpitch:0.0f}\n'.format(xpitch=_map.xpitch*1e3))
            file.write('PITCH_Y={ypitch:0.0f}\n'.format(ypitch=_map.ypitch*1e3))
            file.write('GRID_X={xgrid:d}\n'.format(xgrid=_map.xgrid))
            file.write('GRID_Y={ygrid:d}\n'.format(ygrid=_map.ygrid))
            file.write('ORIGIN_X={xorigin:0.0f}\n'\
                    .format(xorigin=_map.data.x[0]*1e3))
            file.write('ORIGIN_Y={yorigin:0.0f}\n'\
                    .format(yorigin=_map.data.y[0]*1e3))
            file.write('MAP\n')
            for i in range(0, len(_map.data.x)):
                    file.write('{xmap:0.2f}\n'.format(xmap=_map.data.xdev[i]*1e3))
            for i in range(0, len(_map.data.y)):
                    file.write('{ymap:0.2f}\n'.format(ymap=_map.data.ydev[i]*1e3))
        return _map

    def upsidedown(self):
        tmp = copy.deepcopy(self)
        tmp.data.x *= -1
        tmp.data.y *= -1
        tmp.data.xdev *= -1
        tmp.data.ydev *= -1
        tmp.data.sort_values(by=['y', 'x'],
                             ascending=[False, True], inplace=True)
        tmp.data.reset_index(inplace=True, drop=True)
        return tmp

    def interpolation(self, x, y):
        pass


    def applycoef(self, order=3, unit='nm'):
        gmc = self.report_gmc(order=order)
        return self + gmc.gmc2regi(grid=self.xgrid, pitch=self.xpitch,
                                   unit=unit)

    def applymap(self):
        pass


def multiplot(df=(), ref=None, scale=5, normalized=False, shift=True):
    """
    df: dataframe of mapwin data
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.plot(df[0].data.x, df[0].data.y, '+', color='gray')
    color_idx = np.linspace(0, 1, len(df))
    legend_list = ['grid 1div: {0:.0f}[nm]'.format(scale)]
    for i, mw in zip(color_idx, df):
        print('plotting {0:s}'.format(mw.name))
        if ref is not None:
            mw = mw-ref
        xgrid = mw.data.x.unique()
        ygrid = mw.data.y.unique()
        legend_list.append(mw.name)

        for x in xgrid:
            if shift:
                ax1.plot((mw.data[mw.data.x == x].xdev - mw.xmean)*1000/scale
                        * mw.xpitch + mw.data[mw.data.x == x].x,
                        (mw.data[mw.data.x == x].ydev - mw.ymean)*1000/scale
                        * mw.ypitch + mw.data[mw.data.x == x].y,
                        'b-', alpha=0.8, color=plt.cm.rainbow(i))
                        #'b-', alpha=0.8, color=plt.cm.rainbow(i*2))
            elif shift is False:
                ax1.plot(mw.data[mw.data.x == x].xdev * 1000 / scale
                        * mw.xpitch + mw.data[mw.data.x == x].x,
                        mw.data[mw.data.x == x].ydev * 1000 / scale
                        * mw.ypitch + mw.data[mw.data.x == x].y,
                        'b-', alpha=0.8, color=plt.cm.rainbow(i))
        for y in ygrid:
            if shift:
                ax1.plot((mw.data[mw.data.y == y].xdev - mw.xmean)
                    * 1000 / scale * mw.xpitch
                    + mw.data[mw.data.y == y].x,
                    (mw.data[mw.data.y == y].ydev - mw.ymean)
                    * 1000 / scale * mw.ypitch
                    + mw.data[mw.data.y == y].y,
                    'b-', alpha=0.8, color=plt.cm.rainbow(i))
                    #'b-', alpha=0.8, color=plt.cm.rainbow(i*2))
            elif shift is False:
                ax1.plot(mw.data[mw.data.y == y].xdev * 1000 / scale
                    * mw.xpitch + mw.data[mw.data.y == y].x,
                    mw.data[mw.data.y == y].ydev * 1000 / scale
                    * mw.ypitch + mw.data[mw.data.y == y].y,
                    'b-', alpha=0.8, color=plt.cm.rainbow(i))

    box = ax1.get_position()
    ax1.legend(legend_list, loc = 'upper left', bbox_to_anchor=(1.05, 1.0),
               fancybox=True, shadow=True, fontsize=9, borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    legend = ax1.get_legend()

    num = len(legend.legendHandles)
    for color, i in zip(color_idx, range(0, num-1)):
    #for color, i in zip(color_idx, range(0, num)):
        #legend.legendHandles[i].set_color(plt.cm.rainbow(color))
        legend.legendHandles[i+1].set_color(plt.cm.rainbow(color))
    ax1.set_xticks([], minor=False)
    ax1.set_yticks([], minor=False)
    ax1.grid(True)
    plt.show(block = False)


if __name__ == '__main__':
    pass
