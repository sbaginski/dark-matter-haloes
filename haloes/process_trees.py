import argparse
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path

from haloes.toolset.halo_reader import HaloReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='color_trees',
        help='Path to the directory with merger tree files to process'
    )
    args = parser.parse_args()

    Path('output/position').mkdir(parents=True, exist_ok=True)
    Path('output/v_circ').mkdir(parents=True, exist_ok=True)
    color_trees = [
        os.path.join(args.input, filename)
        for filename in os.listdir(args.input)
    ]
    color_trees.sort()
    with Pool(processes=min(len(color_trees), cpu_count())) as pool:
        processed = pool.map(HaloReader.process_tree, color_trees)
        print('Plotting graphs...')
        _ = pool.map(HaloReader.plot_tree, processed)


if __name__ == '__main__':
    main()
