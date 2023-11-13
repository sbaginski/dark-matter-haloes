import argparse
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from haloes.toolset.halo_reader import HaloReader


def map_position(x, y, z, grid_size, total_length=70.4):
    cell_size = total_length / grid_size
    x_index = (x // cell_size).astype(int)
    y_index = (y // cell_size).astype(int)
    z_index = (z // cell_size).astype(int)
    return x_index, y_index, z_index


def run():
    flag_ids = {0: 'Void', 1: 'Undefined',
                2: 'Wall', 3: 'Filament', 4: 'Node'}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--color_trees',
        type=str,
        default='color_trees',
        help='Path to the directory with merger tree files to process'
    )
    parser.add_argument(
        '-n',
        '--nexus',
        type=str,
        default='color_cold_s159_nexus_all_clean.MMF',
        help='Path to the file with NEXUS+ classification output'
    )
    args = parser.parse_args()
    Path(os.path.join('output', 'cross_check')).mkdir(
        parents=True, exist_ok=True)

    nexus_voxels = HaloReader.read_nexus(args.nexus)

    color_trees = [
        os.path.join(args.color_trees, filename)
        for filename in os.listdir(args.color_trees)
    ]
    with Pool(processes=min(len(color_trees), cpu_count())) as pool:
        processed = pool.map(HaloReader.process_tree, color_trees)

    massive = pd.DataFrame()
    for entry in tqdm(processed):
        haloes = entry[0]
        filename = entry[1]
        if haloes.empty:
            continue
        haloes = haloes.sort_values(by=['mass'], ascending=False)
        haloes = haloes.head(100)
        massive = pd.concat([massive, haloes])

        x_ind, y_ind, z_ind = map_position(
            np.array(haloes['position_x']),
            np.array(haloes['position_y']),
            np.array(haloes['position_z']),
            np.shape(nexus_voxels)[0]
        )
        nexus_values = nexus_voxels[x_ind, y_ind, z_ind]
        haloes['type'] = [flag_ids[value] for value in nexus_values]

        fig, ax = plt.subplots()
        ax.hist(haloes['type'], bins=5)
        fig.tight_layout()
        output_filename = os.path.basename(filename).replace('.hdf5', '.pdf')
        plt.savefig(os.path.join('output', 'cross_check',
                    output_filename), format='pdf')
        plt.close()

        fig, ax = plt.subplots()
        logbins = np.logspace(
            np.log10(np.min(haloes['mass'])), np.log10(np.max(haloes['mass'])), 10)
        ax.hist(haloes['mass'], bins=logbins)
        plt.xscale('log')
        fig.tight_layout()
        output_filename = os.path.basename(
            filename).replace('.hdf5', '_mass.pdf')
        plt.savefig(os.path.join('output', 'cross_check',
                    output_filename), format='pdf')
        plt.close()

    massive = massive.sort_values(by=['mass'], ascending=False)
    massive = massive.head(100)
    x_ind, y_ind, z_ind = map_position(
        np.array(massive['position_x']),
        np.array(massive['position_y']),
        np.array(massive['position_z']),
        np.shape(nexus_voxels)[0]
    )
    nexus_values = nexus_voxels[x_ind, y_ind, z_ind]
    massive['type'] = [flag_ids[value] for value in nexus_values]

    fig, ax = plt.subplots()
    ax.hist(massive['type'], bins=5)
    fig.tight_layout()
    output_filename = 'massive.pdf'
    plt.savefig(os.path.join('output', 'cross_check',
                output_filename), format='pdf')
    plt.close()

    fig, ax = plt.subplots()
    logbins = np.logspace(np.log10(np.min(massive['mass'])), np.log10(
        np.max(massive['mass'])), 10)
    ax.hist(massive['mass'], bins=logbins)
    plt.xscale('log')
    fig.tight_layout()
    output_filename = 'massive_mass.pdf'
    plt.savefig(os.path.join('output', 'cross_check',
                output_filename), format='pdf')
    plt.close()


if __name__ == '__main__':
    run()
