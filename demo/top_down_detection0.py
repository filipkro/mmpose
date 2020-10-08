import os
from argparse import ArgumentParser

import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

import time
import numpy as np


def box_check(img, device='cpu'):
    flip = False
    det_config = '/home/filipkr/Documents/xjob/mmpose/mmdetection/' +\
        'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    det_model = '/home/filipkr/Documents/xjob/mmpose/mmdetection/' +\
        'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    det_model = init_detector(det_config, det_model, device=device)
    print('loaded detection model')

    det_results = inference_detector(det_model, img)
    # bbox = det_results[0]
    bbox = np.expand_dims(np.array(det_results[0])[0, :], axis=0)
    bbox[0, 2:4] = bbox[0, 2:4] + 100
    # print(bbox)
    if abs(bbox[0, 0] - bbox[0, 2]) > abs(bbox[0, 1] - bbox[0, 3]):
        flip = True
        bbox[0, 1] -= 100
        bbox = [[bbox[0, 1], bbox[0, 0], bbox[0, 3], bbox[0, 2], bbox[0, 4]]]
        print('frames will be flipped')
    else:
        bbox[0, 0] -= 100

    print('bounding box found: {0}'.format(bbox))
    show_result_pyplot(det_model, img, det_results)

    return bbox, flip


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('det_config', help='Config file for detection')
    # parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument('--show', action='store_true', default=False,
                        help='whether to show visualizations.')
    parser.add_argument('--out-video-root', default='',
                        help='Root of the output video file. '
                        'Default not saving the visualization video.')
    parser.add_argument('--device', default='cpu',
                        help='Device used for inference')
    parser.add_argument('--bbox-thr', type=float, default=0.3,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='Keypoint score threshold')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--only_box', type=bool, default=False)
    # parser.add_argument('--csv-path', type=str, help='CSV path')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # assert args.det_config is not None
    # assert args.det_checkpoint is not None

    # build the pose model from a config file and a checkpoint file

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 device=args.device)
    print('loaded pose model')

    dataset = pose_model.cfg.data['test']['type']

    print(dataset)

    mod_used = pose_model.cfg.model['backbone']['type']

    print('model used {0}'.format(mod_used))

    cap = cv2.VideoCapture(args.video_path)
    print('loaded video...')
    print('checking orientation and position')

    flag, img = cap.read()
    cap.release()
    person_bboxes, flip = box_check(img)
    cap = cv2.VideoCapture(args.video_path)

    print(args.only_box)
    if args.only_box:
        # cv2.waitKey(0)
        return

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True
        print('save path: {0}'.format(args.out_video_root))

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if flip:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        else:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        m_dim = max(size)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.file_name == '':
            fname = os.path.join(args.out_video_root,
                                 f'vis_{os.path.basename(args.video_path)}')
            # if os.path.basename(args.video_path).find()
            fname = fname.replace(fname[fname.find('.', -5)::], '')
            fname += mod_used + dataset + '.mp4'
            print('FN {0}'.format(fname))
            while os.path.isfile(fname):
                fname = fname.replace('.mp4', '')

                idx = fname.find('-', -4)
                if idx == -1:
                    fname += '-0.mp4'
                else:
                    fname = fname.replace(fname[idx + 1::],
                                          str(int(fname[idx + 1::])
                                              + 1) + '.mp4')
        else:
            fname = os.path.join(args.out_video_root, args.file_name)

        print(fname)
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, size)

    print(pose_model.cfg.channel_cfg['num_output_channels'])
    poses = np.zeros((frames,
                      pose_model.cfg.channel_cfg['num_output_channels'], 3))
    # poses[-1, 0:2] = size
    print(poses.shape)

    frame = 0
    t0 = time.perf_counter()
    prev_pose = 0

    width = (cap.get(3))
    height = (cap.get(4))

    print('width: {0}, height: {1}'.format(width, height))

    skip_ratio = 1

    # person_bboxes = [[2 * width / 10, height /
    #                   8, 0.9 * width, 7 * height / 8, 1]]

    # person_bboxes = [[2 * width / 10, height /
    #                   5, 0.9 * width, 4 * height / 5, 1]]
    # person_bboxes = [[2*width/10, 0, 0.9*width, height, 1]]
    # person_bboxes = [[3 * width / 10, 0, 0.6 * width, height, 1]]
    # person_bboxes = [[35 * width / 10, 0.1 *
    #                   height, 0.7 * width, 0.95 * height, 1]]
    print(person_bboxes)
    # rmin = np.ones(2)
    # rmax = np.zeros(2)
    # lmin = np.ones(2)
    # lmax = np.zeros(2)
    lmin = 1
    lmax = 0
    rmin = 1
    rmax = 0
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()
        if flip:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if not flag:
            break

        # check every 2nd frame
        if frame % skip_ratio == 0:
            # test a single image, the resulting box is (x1, y1, x2, y2)
            # det_results = inference_detector(det_model, img)
            # # keep the person class bounding boxes.
            #
            # person_bboxes = np.expand_dims(
            #     np.array(det_results[0])[0, :], axis=0)
            #
            # print(person_bboxes)

            # test a single image, with a list of bboxes.
            pose_results = inference_top_down_pose_model(pose_model, img,
                                                         person_bboxes,
                                                         bbox_thr=args.bbox_thr,
                                                         format='xyxy',
                                                         dataset=dataset)

            t = time.perf_counter()
            print('Frame {0} out of {3} analysed in {1} secs. Total time: {2} secs\
                    '.format(frame, t - t1, t - t0, frames))

            # show the results
            if np.shape(pose_results)[0] > 0:
                prev_pose = pose_results
                # x_ratios = pose_results[0]['keypoints'][:, 0] / m_dim
                # y_ratios = pose_results[0]['keypoints'][:, 1] / m_dim
                ratios = pose_results[0]['keypoints'][:, 0:2] / m_dim

                lmin = min((ratios[13, 1], lmin))
                lmax = max((ratios[13, 1], lmax))
                rmin = min((ratios[14, 0], rmin))
                rmax = max((ratios[14, 1], rmax))
                # lmin[0] = min((ratios[13, 0], lmin[0]))
                # lmin[1] = min((ratios[13, 1], lmin[1]))
                # lmax[0] = max((ratios[13, 0], lmax[0]))
                # lmax[1] = max((ratios[13, 1], lmax[1]))
                #
                # rmin[0] = min((ratios[14, 0], rmin[0]))
                # rmin[1] = min((ratios[14, 1], rmin[1]))
                # rmax[0] = max((ratios[14, 0], rmax[0]))
                # rmax[1] = max((ratios[14, 1], rmax[1]))

                if (rmax - rmin) > 0.1 or (frame > 150 and
                                           (rmax - rmin) > (lmax - lmin)):

                poses[frame, ...] = ratios
                # poses[frame, :, 0] = x_ratios
                # poses[frame, :, 1] = y_ratios
                # poses[frame, :, 0] = pose_results[0]['keypoints'][:, 0] / m_dim
                # poses[frame, :, 1] = pose_results[0]['keypoints'][:, 1] / m_dim

            else:
                pose_results = prev_pose  # or maybe just skip saving
                print('lol')

        else:
            pose_results = prev_pose

        vis_img = vis_pose_result(pose_model, img, pose_results,
                                  dataset=dataset, kpt_score_thr=args.kpt_thr,
                                  show=False)

        if args.show or frame % skip_ratio == 0:
            cv2.imshow('Image', vis_img)
        frame += 1

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
        out_file = fname.replace('.mp4', '.npy')
        np.save(out_file, poses)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('starting...')
    main()
