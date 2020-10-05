import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import filt


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    args = parser.parse_args()
    poses = np.load(args.np_file)

    poses = poses[6:-1, :, 0:2]
    poses_pre_process = np.copy(poses)
    # if poses[0, 0, 0] < 1:
    #     poses[..., 0] = poses[..., 0] * 1920
    #     poses[..., 1] = poses[..., 1] * 1080
    poses = filt.outliers(poses)
    poses = filt.smoothing(poses)
    poses = filt.fix_hip(poses)

    print(poses_pre_process - poses)
    plt.plot(poses_pre_process[:, 14, 0])
    plt.plot(poses[:, 14, 0])
    plt.show()
    # dhip = abs(poses[:, 11, 1] - poses[:, 12, 1])
    # hip_mean = np.mean(dhip) * np.ones(dhip.shape)
    # hip_med = np.median(dhip) + np.ones(dhip.shape)
    # plt.plot(dhip)
    # plt.plot(hip_mean)
    # plt.plot(hip_med)
    # plt.show()


if __name__ == '__main__':
    main()
