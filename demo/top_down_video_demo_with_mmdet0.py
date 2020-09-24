import os
from argparse import ArgumentParser

import cv2
from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

import time


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    print('loaded detection model')
    # build the pose model from a config file and a checkpoint file
    print('pose config: {0} \npose checkpoint: {1}'.format(args.pose_config, args.pose_checkpoint))
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)
    print('loaded poes model')

    dataset = pose_model.cfg.data['test']['type']

    print(dataset)

    cap = cv2.VideoCapture(args.video_path)

    print('loaded video')

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
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    count = 0
    t0 = time.perf_counter()
    while (cap.isOpened()):
        t1 = time.perf_counter()
        flag, img = cap.read()

        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        det_results = inference_detector(det_model, img)
        # keep the person class bounding boxes.
        person_bboxes = det_results[0].copy()


        # test a single image, with a list of bboxes.
        pose_results = inference_top_down_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset)

        count += 1
        t = time.perf_counter()
        print('Frame {0} analysed in {1} secs. Total time: {2} secs\
                '.format(count, t - t1, t - t0))

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show or count == 3:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('starting...')
    main()
