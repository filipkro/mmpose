import numpy as np
# import filt
from argparse import ArgumentParser
from split_sequence import split
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def parse_mts(poses, kpts, label, mts=None, mts_labels=None, angles=None):

    split_poses, _ = split(poses)
    print('split', split_poses.shape)

    if angles is not None:
        ang = calc_angle(split_poses, angles[0], angles[1])

    kpts.append(14)
    split_poses = split_poses[:, :, kpts, :]

    sh = split_poses.shape
    split_poses = split_poses.reshape((sh[0], sh[1], sh[2] * sh[3]))

    if angles is not None:
        xtra = split_poses[:, :, -1]
        split_poses = np.append(split_poses[:, :, :-1], ang, axis=2)
        split_poses = np.append(split_poses,
                                np.expand_dims(xtra, axis=-1), axis=2)

    assert mts.shape[2] == split_poses.shape[2]

    if mts is not None:
        print('mts', mts.shape)
        min_len = np.min((mts.shape[1], split_poses.shape[1]))
        print(min_len)
        if min_len < split_poses.shape[1]:
            # print(peaks)
            crop = np.zeros(mts.shape)
            p_len = split_poses.shape[1]
            for i in range(split_poses.shape[0]):
                peak, _ = find_peaks(split_poses[i, :, -1], distance=40,
                                     prominence=0.05, width=10)
                peak = peak[0]
                if ((peak >= int(min_len / 2)) and
                        (peak <= p_len - int(min_len / 2))):
                    crop[i, ...] = split_poses[i, peak - int(min_len / 2):
                                               peak +
                                               int(np.ceil(min_len / 2)), ...]
                elif peak < p_len / 2:
                    crop[i, ...] = split_poses[i, :min_len, ...]
                else:
                    crop[i, ...] = split_poses[i, -min_len:, ...]

            split_poses = crop

        elif min_len < mts.shape[1]:
            crop = np.zeros(split_poses.shape)
            p_len = mts.shape[1]
            for i in range(mts.shape[0]):
                peak, _ = find_peaks(mts[i, :, -1], distance=40,
                                     prominence=0.05, width=10)
                peak = peak[0]
                if ((peak >= int(min_len / 2)) and
                        (peak <= p_len - int(min_len / 2))):
                    crop[i, ...] = mts[i, peak - int(min_len / 2):
                                       peak + int(np.ceil(min_len / 2)), ...]
                elif peak < p_len / 2:
                    crop[i, ...] = mts[i, :min_len, ...]
                else:
                    crop[i, ...] = mts[i, -min_len:, ...]

            mts = crop

        mts = np.append(mts, split_poses, axis=0)
        mts_labels = np.append(mts_labels, label)

    else:
        mts = split_poses
        mts_labels = label

    print(mts.shape)
    plt.plot(mts[:, :, -1].T)
    plt.show()


def calc_angle(poses, kpt1, kpt2):
    angles = np.arctan2(poses[..., kpt1, 1] - poses[..., kpt2, 1],
                        poses[..., kpt1, 0] - poses[..., kpt2, 0])
    print('dims in calac, {0}, {1}'.format(poses.shape, angles.shape))

    return np.expand_dims(angles, axis=-1)


def fake_mts(file, kpts, angles=False):
    mts = np.load(file)
    mts = mts[6:-1, :, 0:2]
    mts, _ = split(mts)
    if angles:
        ang = calc_angle(mts, 11, 13)
    kpts.append(14)
    mts = mts[:, :, kpts, :]
    sh = mts.shape
    mts = mts.reshape((sh[0], sh[1], sh[2] * sh[3]))

    if angles:
        xtra = mts[:, :, -1]
        mts = np.append(mts[:, :, :-1], ang, axis=2)
        mts = np.append(mts, np.expand_dims(xtra, axis=-1), axis=2)
        # mts = np.insert(mts, -1, ang, axis=2)

    # plt.plot(mts[..., -1].T)
    # plt.show()

    return mts


def main():
    parser = ArgumentParser()
    parser.add_argument('np_file', help='Numpy file to filter')
    parser.add_argument('np_file2', default='', help='Numpy file to filter')
    args = parser.parse_args()

    poses = np.load(args.np_file)

    poses = poses[6:-1, :, 0:2]
    kpts = [12, 13, 15]
    if args.np_file2 != '':
        # ang = calc_angle(poses, 11, 13)
        mts = fake_mts(args.np_file2, kpts.copy(), angles=True)
        parse_mts(poses, kpts, 1, mts=mts,
                  mts_labels=np.ones(mts.shape[1]), angles=(11, 13))
    else:
        parse_mts(poses, kpts, 1)


if __name__ == '__main__':
    main()
