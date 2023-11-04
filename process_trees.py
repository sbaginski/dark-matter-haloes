import argparse
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path

from halo_reader import HaloReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='color_trees',
        help='Path to the directory with merger tree files to process'
    )
    args = parser.parse_args()

    Path('processed').mkdir(parents=True, exist_ok=True)
    Path('output/position').mkdir(parents=True, exist_ok=True)
    Path('output/v_circ').mkdir(parents=True, exist_ok=True)
    color_trees = [
        os.path.join(args.input_dir, filename)
        for filename in os.listdir(args.input_dir)
    ]
    processed = [
        os.path.join('processed', filename)
        for filename in os.listdir('processed')
    ]
    to_process = [filepath for filepath in color_trees
                  if os.path.join(
                      'processed',
                      os.path.basename(filepath).replace('.hdf5', '.csv')
                  ) not in processed]
    color_trees.sort()
    with Pool(processes=min(len(color_trees), cpu_count())) as pool:
        processed += pool.map(HaloReader.process_tree, to_process)
        print('Plotting graphs...')
        _ = pool.map(HaloReader.plot_tree, processed)


if __name__ == '__main__':
    main()
