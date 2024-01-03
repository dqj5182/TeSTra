# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right
import time
import pickle as pkl

import torch
import numpy as np

from rekognition_online_action_detection.evaluation import compute_result

from ..base_inferences.perframe_det_batch_inference import do_perframe_det_batch_inference


from ..engines import INFERENCES as registry


@registry.register('LSTR')
def do_lstr_batch_inference(cfg,
                            model,
                            device,
                            logger):
    if cfg.MODEL.LSTR.INFERENCE_MODE == 'stream':
        do_lstr_stream_inference(cfg,
                                 model,
                                 device,
                                 logger)


def do_lstr_stream_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    # Collect scores and targets
    pred_scores = []
    gt_targets = []

    def to_device(x, dtype=np.float32):
        return torch.as_tensor(x.astype(dtype)).unsqueeze(0).to(device)

    long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
    long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
    work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE

    pred_scores_all = {}
    gt_targets_all = {}

    with torch.no_grad():
        for session_idx, session in enumerate(cfg.DATA.TEST_SESSION_SET): # cfg.DATA.TEST_SESSION_SET is list of videos
            model.clear_cache()

            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                pred_scores = [[] for _ in range(cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES)]
            else:
                pred_scores = []
            gt_targets = []

            visual_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.VISUAL_FEATURE, session + '.npy'), mmap_mode='r')
            motion_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.MOTION_FEATURE, session + '.npy'), mmap_mode='r')
            object_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.OBJECT_FEATURE, session + '.npy'), mmap_mode='r')
            target = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME, session + '.npy'))

            start_time = time.time()

            for work_start, work_end in zip(range(0, target.shape[0] + 1),
                                            range(work_memory_length, target.shape[0] + 1)):
                # Get target
                # target = target[::work_memory_sample_rate]

                # Get work memory
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::work_memory_sample_rate]
                work_visual_inputs = to_device(visual_inputs[work_indices])
                work_motion_inputs = to_device(motion_inputs[work_indices])
                work_object_inputs = to_device(object_inputs[work_indices])

                # Get long memory
                PRECISE = True
                if PRECISE:
                    long_end = work_start - long_memory_sample_rate
                    if long_end < 0:
                        long_indices = [0 for _ in range(long_memory_num_samples)]
                        long_indices_set = [long_indices for _ in range(long_memory_sample_rate)]
                        long_visual_inputs = to_device(visual_inputs[long_indices])
                        long_motion_inputs = to_device(motion_inputs[long_indices])
                        long_object_inputs = to_device(object_inputs[long_indices])
                    else:
                        long_indices = long_indices_set[long_end % long_memory_sample_rate][1:] + [long_end]
                        long_indices_set[long_end % long_memory_sample_rate] = long_indices
                        long_visual_inputs = to_device(visual_inputs[[long_end]])
                        long_motion_inputs = to_device(motion_inputs[[long_end]])
                        long_object_inputs = to_device(object_inputs[[long_end]])
                else:
                    long_end = work_start - 1
                    if long_end == -1:
                        long_indices = [0 for _ in range(long_memory_num_samples)]
                        long_visual_inputs = to_device(visual_inputs[long_indices])
                        long_motion_inputs = to_device(motion_inputs[long_indices])
                        long_object_inputs = to_device(object_inputs[long_indices])
                    elif long_end % long_memory_sample_rate == 0:
                        long_indices = long_indices[1:] + [long_end]
                        long_visual_inputs = to_device(visual_inputs[[long_end]])
                        long_motion_inputs = to_device(motion_inputs[[long_end]])
                        long_object_inputs = to_device(object_inputs[[long_end]])
                    else:
                        long_visual_inputs = None
                        long_motion_inputs = None
                        long_object_inputs = None

                # Get memory key padding mask
                memory_key_padding_mask = np.zeros(len(long_indices))
                last_zero = bisect_right(long_indices, 0) - 1
                if last_zero > 0:
                    memory_key_padding_mask[:last_zero] = float('-inf')
                memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0).to(device)

                score = model.stream_inference(
                    long_visual_inputs,
                    long_motion_inputs,
                    long_object_inputs,
                    work_visual_inputs,
                    work_motion_inputs,
                    work_object_inputs,
                    memory_key_padding_mask,
                    cache_num=long_memory_sample_rate if PRECISE else 1,
                    cache_id=long_end % long_memory_sample_rate if PRECISE else 0)[0]

                if cfg.DATA.DATA_NAME.startswith('EK'):
                    score = score.cpu().numpy()
                else:
                    score = score.softmax(dim=-1).cpu().numpy()

                if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                    if work_start == 0:
                        upsample_score = score[:-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES].repeat(work_memory_sample_rate, axis=0)
                        upsample_score = upsample_score[work_start:work_end-1]
                        anticipate_score = score[-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]
                        # coarse but sufficient
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            combined_score = np.concatenate((upsample_score, anticipate_score[:t_a + 1]),
                                                            axis=0)
                            pred_scores[t_a].extend(list(combined_score))
                        gt_targets.extend(list(target[:work_end-1]))
                    else:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            pred_scores[t_a].append(list(score[t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]))
                        gt_targets.append(list(target[work_end - 1]))
                else:
                    if work_start == 0:
                        gt_targets.extend(list(target[:work_end]))
                        pred_scores.extend(list(score))
                    else:
                        gt_targets.append(list(target[work_end - 1]))
                        pred_scores.append(list(score[-1]))

            end_time = time.time()
            logger.info('Running time: {:.3f} seconds'.format(end_time - start_time))

            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                    result = compute_result['perframe'](
                        cfg,
                        gt_targets,
                        pred_scores[t_a][:-1 - t_a],
                    )
                    sec = (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
                    logger.info('mAP of video ({:.2f}s) {}: {:.5f}'.format(sec, session, result['mean_AP']))
            else:
                result = compute_result['perframe'](
                    cfg,
                    gt_targets,
                    pred_scores,
                )
                logger.info('mAP of video {}: {:.5f}'.format(session, result['mean_AP']))


            gt_targets_all[session] = np.array(gt_targets)
            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                    pred_scores[t_a] = np.array(pred_scores[t_a][:- 1 - t_a])
                pred_scores_all[session] = np.stack(pred_scores, axis=0).transpose(1, 2, 0)
                pred_scores_all[session] = pred_scores_all[session]
            else:
                pred_scores_all[session] = np.array(pred_scores)


    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores_all,
    #     'perframe_gt_targets': gt_targets_all,
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.stream.pkl', 'wb'))

    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        maps_list = []
        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
            result = compute_result['perframe'](
                cfg,
                np.concatenate(list(gt_targets_all.values()), axis=0),
                np.concatenate(list(pred_scores_all.values()), axis=0)[:, :, t_a],
            )
            logger.info('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
                (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                cfg.DATA.METRICS, result['mean_AP']
            ))
            maps_list.append(result['mean_AP'])
        logger.info('Action anticipation (mean) perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, np.mean(maps_list)
        ))
    else:
        result = compute_result['perframe'](
            cfg,
            np.concatenate(list(gt_targets_all.values()), axis=0),
            np.concatenate(list(pred_scores_all.values()), axis=0),
        )
        logger.info('Action detection perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, result['mean_AP']
        ))

    gt_targets = gt_targets_all
    pred_scores = pred_scores_all