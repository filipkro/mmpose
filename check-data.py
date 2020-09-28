import numpy as np
import matplotlib.pyplot as plt


nf = 2
filt = np.empty(2 * nf + 1)


# def fix_outlier(idx, dim):
#     return (kpt[idx - 1, dim] + kpt[idx + 1, dim]) / 2
#     # return kpt[idx - 1, dim]

def fix_outlier(idx, dim):
    return (dim[idx - 1] + dim[idx + 1]) / 2
    # return kpt[idx - 1, dim]


def init_weights():
    global filt
    structure = np.ones(nf + 1)
    # structure = np.arange(nf + 1) + 1
    filt = np.convolve(structure, np.flip(structure))
    filt = np.ones(2 * nf + 1)
    filt = filt / np.sum(filt)

    print(filt)
    print(np.sum(filt))


def filter(idx, x_coords, y_coords):
    print(filt)
    print(x_coords[idx - nf:idx + nf + 1])
    x = np.sum(x_coords[idx - nf:idx + nf + 1] * filt)
    y = np.sum(y_coords[idx - nf:idx + nf + 1] * filt)
    # print(x - x_coords[idx])
    x = x_coords[idx]
    y = y_coords[idx]

    return (x, y)


PATH = '/home/filipkr/Documents/xjob/vids/out/'
FP = PATH + 'vis_001FLHRNetTopDownCocoDataset-2.npy'

poses = np.load(FP)

if poses[0, 0, 0] < 1:
    poses[..., 0] = poses[..., 0] * 1920
    poses[..., 1] = poses[..., 1] * 1080

kpt = poses[:, 14, 0:2]

frames = kpt.shape[0]
print(frames)

print(kpt.shape)
# plt.plot(kpt)
# plt.show()
xc = np.zeros((frames, 1))
yc = np.zeros((frames, 1))


# for i in range(1, frames - 1):
#     xc[i:i + 1] = kpt[i:1 + 1, 0]
#     yc[i:i + 1] = kpt[i:i + 1, 1]
#     xc[i] = fix_outlier(i, xc) if (
#         abs(kpt[i, 0] - kpt[i - 1, 0]) > 0.01) else kpt[i, 0]
#     yc[i] = fix_outlier(i, yc) if (
#         abs(kpt[i, 1] - kpt[i - 1, 1]) > 0.01) else kpt[i, 1]
xc[0] = kpt[0, 0]
yc[0] = kpt[0, 1]
for a in range(1, 4):
    for i in range(a, frames - a):
        x = kpt[i, 0]
        y = kpt[i, 1]

        if abs(x - xc[i - 1]) > 0.01:
            x = (xc[i - 1] + kpt[i + 1, 0]) / 2
        if abs(y - yc[i - 1]) > 0.01:
            y = (yc[i - 1] + kpt[i + 1, 1]) / 2

        xc[i] = x
        yc[i] = y

xof = xc
yof = yc

init_weights()
# for i in range(nf + 1, frames - 1 - nf):
#     print('x before {0}'.format(xc[i]))
#     xc[i], yc[i] = filter(i, xc, yc)
#     print('x after {0}'.format(xc[i]))

# plt.plot(xc)
# plt.plot(xof)
plt.plot(kpt[:, 0])
# plt.plot(yc)
# plt.plot(yof)
plt.plot(kpt[:, 1])
plt.show()
#
#
# for x, y in kpt:
#     print(x, y)


#
#
# def main():
#
# if __name__ == '__main__':
#     main()
