import argparse
import os
from pathlib import Path

from tqdm import tqdm

from toolset.halo_reader import HaloReader


def main():
    INPUT_PATH = './color_cold_s159_nexus_all_clean.MMF'
    parser = argparse.ArgumentParser(
        prog='top',
        description='Visualise NEXUS+ classification output'
    )
    parser.add_argument('-i', '--input', type=str, default=INPUT_PATH)
    args = parser.parse_args()
    Path(os.path.join('output', 'nexus')).mkdir(parents=True, exist_ok=True)

    nexus_voxels = HaloReader.read_nexus(args.input)
    for slice_index in tqdm([0, 50, 100, 150, 200, 255]):
        HaloReader.plot_flags(
            nexus_voxels,
            slice_index=slice_index,
            output_path=os.path.join('output', 'nexus',
                                     f'nexus_slice_{slice_index}.pdf')
        )


if __name__ == '__main__':
    main()
