import os
from argparse import ArgumentParser

import cv2
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

import time
import numpy as np


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
    # parser.add_argument('--csv-path', type=str, help='CSV path')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # assert args.det_config is not None
    # assert args.det_checkpoint is not None

    # det_model = init_detector(
    #     args.det_config, args.det_checkpoint, device=args.device)
    # print('loaded detection model')
    # build the pose model from a config file and a checkpoint file

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 device=args.device)
    print('loaded poes model')

    dataset = pose_model.cfg.data['test']['type']

    print(dataset)

    mod_used = pose_model.cfg.model['backbone']['type']

    print('model used {0}'.format(mod_used))

    cap = cv2.VideoCapture(args.video_path)

    print('loaded video')

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True
        print('save path: {0}'.format(args.out_video_root))

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fname = os.path.join(args.out_video_root,
                             f'vis_{os.path.basename(args.video_path)}')

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
        print(fname)
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, size)

    print(pose_model.cfg.channel_cfg['num_output_channels'])
    poses = np.zeros((frames,
                      pose_model.cfg.channel_cfg['num_output_channels'], 3))

    print(poses.shape)

    frame = 0
    t0 = time.perf_counter()
    prev_pose = 0

    width = (cap.get(3))
    height = (cap.get(4))

    print('width: {0}, height: {1}'.format(width, height))

    skip_ratio = 1

    person_bboxes = [[2 * width / 10, height /
                      8, 0.9 * width, 7 * height / 8, 1]]

    person_bboxes = [[2 * width / 10, height /
                      5, 0.9 * width, 4 * height / 5, 1]]
    # person_bboxes = [[0, 0, width, height, 1]]
    print(person_bboxes)
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()

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

            # print(pose_results)
            # np_results = np.asarray(pose_results[0]['keypoints'])
            # print(pose_results[0]['keypoints'])
            # print('Result shape: {0}'.format(np_results.shape))

            # show the results
            if np.shape(pose_results)[0] > 0:
                prev_pose = pose_results
                # x_ratios = pose_results[0]['keypoints'][:, 0] / width
                # y_ratios = pose_results[0]['keypoints'][:, 1] / height
                # poses[frame, :, 0] = x_ratios
                # poses[frame, :, 1] = y_ratios
                poses[frame, :, 0] = pose_results[0]['keypoints'][:, 0]
                poses[frame, :, 1] = pose_results[0]['keypoints'][:, 1]

            else:
                pose_results = prev_pose  # or maybe just skip saving
                print('lol')

            # print('bbox shape: {0}'.format(np.array(person_bboxes)))
            #
            # print('pose_result shape: {0}'.format(
            #     np.array(pose_results).shape))
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
        else:
            pose_results = prev_pose

        vis_img = vis_pose_result(pose_model, img, pose_results,
                                  dataset=dataset, kpt_score_thr=args.kpt_thr,
                                  show=False)

        # if
        # pose_results =

        # poses[frame, ...] = pose_results[0]['keypoints']

        # print(frame)

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
