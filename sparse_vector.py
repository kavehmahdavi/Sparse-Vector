#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

""" Sparse Vector

This is a Sparse Vector class includes:
    - Sparse vector data structure
    - Dot product
    - Convert Array to Sparse Vector


Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
First update: 19/03/2022
Last update: 19/03/2022

Note:

"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from timeit import default_timer as timer
from numpy import array, float64, int32
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import sys

np.random.seed(4)


def progress_bar(count, total, status='', bar_len=100, functionality=None):
    """ Shows and updates progress bar.

    Args:
        bar_len (int): Indicates the progress bar length.
        count (int): Indicates the actual value
        total (int): Indicates the overall value
        status (str): Shows the more related information about the progress bar

    Returns:

    """
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r\033[1;36;m[%s] %s%s ...%s - %s' % (bar,percents,'%',status,functionality))
    sys.stdout.flush()


class SVectors(object):
    """
    Base class of the Sparse Vectors.
    """

    def __init__(self, *args):
        """ Initiate a sparse vector,

        Arges:
            args: Input data as a dictionary or a generic 1D numpy array,

        example:
            >>> SVectors({ 5:0.3, 6:2.1 })
            >>> SVectors([0.0,0.0,0.0,0.0,0.0,0.3,2.1])
        """
        self.data = args[0]
        if type(self.data) != dict:
            self.data = self.array2dict(self.data)
        else:
            pass
        self.keys = array(list(self.data.keys()), dtype=int32)
        self.values = array(list(self.data.values()), dtype=float64)

    @staticmethod
    def array2dict(arr):
        idx = np.nonzero(arr)[0]
        return dict(zip(idx, arr[idx]))

    def dot(self, second):
        """ Dot product with the SparseVectors

        Args:
            second: Is the second operand  that can be a SVectors, dictionary or a generic 1D numpy array.

        example:
            >>> sv_object =SVectors({ 5:0.3, 6:2.1 })
            >>> sv_objet.dot(sv_object) / sv_objet.dot([0.0,0.0,0.0,0.0,0.0,0.3,2.1]) / sv_objet.dot({ 5:0.3, 6:2.1 })
                4.5
        """
        _dot = 0.0
        if type(second) in [dict, SVectors]:
            p, s = 0, 0
            while p < len(self.keys) and s < len(second.keys):
                if self.keys[p] < second.keys[s]:
                    p += 1
                elif self.keys[p] > second.keys[s]:
                    s += 1
                else:
                    _dot += self.values[p] * second.values[s]
                    p, s = p + 1, s + 1
            return _dot
        else:
            for p in range(len(self.keys)):
                _dot += self.values[p] * second[self.keys[p]]
            return _dot


def __np_dot(a, b, _print=False):
    start = timer()
    _dot = np.dot(a, b)
    end = timer()
    if _print:
        print("The Sparse_V dot product result is {}. The baseline elapsed time is {} ms".format(_dot,
                                                                                                 (end - start) * 1000))
    return {'BL_Time': (end - start) * 1000, 'BL_Value': _dot}


def __sv_dot(a, b, _print=False):
    start = timer()
    _dot = a.dot(b)
    end = timer()
    if _print:
        print("The Sparse_V dot product result is {}. The baseline elapsed time is {} ms".format(_dot,
                                                                                                 (end - start) * 1000))
    return {'SV_Time': (end - start) * 1000, 'SV_Value': _dot}


def _single_test_sv(n=1000):
    """Simple Empirical Test """
    # Sample generator
    a_1 = sparse.random(1, n, density=0.001)
    a_2 = sparse.random(1, n, density=0.001)

    # Baseline1
    a_2 = a_2.toarray().flatten()
    a_1 = a_1.toarray().flatten()
    __np_dot(a_1, a_1, _print=True)

    # My approach
    a = SVectors(a_1)
    b = SVectors(a_2)
    __sv_dot(a, a, _print=True)


def _test_sv(_rep, _size, _density, _plot=True):
    """ Stochastic Empirical Test """
    # Initiate the progressbar
    _iter = 0
    _iteration = _rep * len(_density) * len(_size)

    result = pd.DataFrame(columns=['Size', 'Density', 'BL_Time', 'BL_Value', 'SV_Time', 'SV_Value'])
    for __density in _density:
        for __size in _size:
            temp = {'Size': __size, 'Density': __density}
            _bl_time, _sv_time = 0.0, 0.0

            for __rep in range(_rep):
                # Generate the sample data
                a, b = sparse.random(1, __size, density=__density), sparse.random(1, __size, density=__density)

                # Baseline
                a, b = a.toarray().flatten(), b.toarray().flatten()
                _bl_temp = __np_dot(a, b)
                _bl_time += _bl_temp.get('BL_Time')

                # My SV approach
                a, b = SVectors(a), SVectors(b)
                _sv_temp = __sv_dot(a, b)
                _sv_time += _sv_temp.get('SV_Time')

                _iter += 1
                progress_bar(_iter, _iteration, bar_len=20,
                             status=str(_iter) + "/" + str(_iteration),
                             functionality='Experiments are conducted')

            temp.update({'BL_Time': _bl_time / _rep,
                         'SV_Time': _sv_time / _rep,
                         'BL_Value': _bl_temp.get('BL_Value'),
                         'SV_Value': _sv_temp.get('SV_Value')})
            result = result.append(temp, ignore_index=True)

    if _plot:
        # Plot the experimental result
        _max = result[['BL_Time', 'SV_Time']].melt().value.max()

        fig = plt.figure(figsize=(16,10), constrained_layout=False)
        fig.suptitle('Empirical Test: SparseVector vs. Numpy Dot', fontsize=14)

        for _index, _item in enumerate(['BL_Time', 'SV_Time']):
            ax = fig.add_subplot(1, 2, _index + 1, projection='3d')

            x, y, z = result['Size'], result['Density'], result[_item]
            plot_x, plot_y = np.meshgrid(np.linspace(np.min(x), np.max(x), 80),
                                         np.linspace(np.min(y), np.max(y), 80))
            plot_z = interp.griddata((x, y), z, (plot_x, plot_y), method='linear')

            surf = ax.plot_surface(plot_x, plot_y, plot_z, cstride=1, rstride=1, cmap='viridis', vmin=0, vmax=_max)
            fig.colorbar(surf, shrink=0.5, aspect=10, orientation="horizontal", pad=0.1)

            # Customize the z axis.
            ax.zaxis.set_major_locator(LinearLocator(10))
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.set_zlim(0, _max)

            ax.set_xlabel('Size')
            ax.set_ylabel('Density')
            ax.set_zlabel('Elapsed Time (ms)')

            if _item is 'BL_Time':
                ax.set_title('Numpy')
            else:
                ax.set_title('SparseVector')

        plt.show()


if __name__ == '__main__':
    _single_test_sv(n=1000)
    rep = 30
    size = range(1000, 25000, 1000)
    density = [0.01, 0.002, 0.005, 0.001, 0.005, 0.0001]
    _test_sv(rep, size, density)
