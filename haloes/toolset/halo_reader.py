import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd


class HaloReader:
    @classmethod
    def read_nexus(cls, filename):
        raw_data = np.fromfile(filename, dtype=np.ushort)
        n_grid = np.uint((raw_data.nbytes / raw_data.itemsize)**(1. / 3.) // 1)
        header_bytes = 1048
        checksum_bytes = 8
        start_idx = int(header_bytes / raw_data.itemsize)
        end_idx = int(checksum_bytes / raw_data.itemsize)
        nexus_voxels = raw_data[start_idx:-
                                end_idx].reshape(n_grid, n_grid, n_grid)
        return nexus_voxels

    @classmethod
    def plot_flags(cls, voxels, slice_index=0, output_path='nexus_output.pdf'):
        flag_ids = {0: 'Void', 1: 'Undefined',
                    2: 'Wall', 3: 'Filament', 4: 'Node'}
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4, clip=True)
        matrix = voxels[slice_index]
        mat = plt.matshow(matrix, norm=norm)

        def lp(i):
            return plt.plot(
                [],
                color=mat.cmap(mat.norm(i)),
                mec="none",
                label=flag_ids[i],
                ls="",
                marker="o"
            )[0]
        handles = [lp(i) for i in flag_ids]
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

    @classmethod
    def process_tree(cls, tree_filename):
        tree_data = h5py.File(tree_filename, 'r')
        node_indices = tree_data['treeIndex/firstNode'][()]

        @njit
        def condition(x, redshift=tree_data['haloTrees/redshift'][()]):
            return redshift[x] == 0

        @njit
        def filter_indices(arr, cond):
            result = np.empty_like(arr)
            j = 0
            for i in range(arr.size):
                if cond(arr[i]):
                    result[j] = arr[i]
                    j += 1
            return result[:j].copy()

        dataset_idx = filter_indices(node_indices, condition)
        position = tree_data['haloTrees/position'][()][dataset_idx]
        haloes = pd.DataFrame(
            {
                'box_size': tree_data['simulation'].attrs['boxSize'],
                'position_x': position[:, 0],
                'position_y': position[:, 1],
                'position_z': position[:, 2],
                'mass': tree_data['haloTrees/nodeMass'][()][dataset_idx],
                'v_circ': tree_data['haloTrees/maximumCircularVelocity'][()][dataset_idx]
            }
        )
        tree_data.close()
        print(f'Processed {tree_filename}')
        return (haloes, tree_filename)

    @classmethod
    def plot_tree(cls, processed_output):
        haloes = processed_output[0]
        halo_filename = processed_output[1]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            haloes['position_x'],
            haloes['position_y'],
            haloes['position_z']
        )
        ax.set_xlabel(r'$x$ [Mpc/h]')
        ax.set_ylabel(r'$y$ [Mpc/h]')
        ax.set_zlabel(r'$z$ [Mpc/h]')
        fig.tight_layout()
        output_path = os.path.join(
            'output',
            'position',
            os.path.basename(halo_filename).replace('.hdf5', '_position.pdf')
        )
        plt.savefig(output_path, format='pdf')
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter(x=haloes['mass'], y=haloes['v_circ'])
        ax.set_xlabel(r'$m$ [Msun/h]')
        ax.set_ylabel(r'$v_{circ}$ [km/s]')
        ax.grid(True)
        fig.tight_layout()
        output_path = os.path.join(
            'output',
            'v_circ',
            os.path.basename(halo_filename).replace('.hdf5', '_v_circ.pdf')
        )
        plt.savefig(output_path, format='pdf')
        plt.close()

        return
