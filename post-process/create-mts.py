import numpy as np
import filt
from argparse import ArgumentParser

KPTS = [6, 8, 11, 12, 14, 16]


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    parser.add_argument('root', help='Folder containing files')
    args = parser.parse_args()

    poses = np.load(args.np_file)
    poses = filt.outliers(poses)
    poses = filt.smoothing(poses)
    poses = filt.fix_hip(poses)

    poses =


if __name__ == '__main__':
    main()
