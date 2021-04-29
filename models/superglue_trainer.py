from copy import deepcopy
from pathlib import Path
import torch
import os
from torch import nn
import shutil
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from torch.autograd import Variable
import matplotlib.cm as cm
from os.path import abspath, dirname
import numpy as np
import random
from pytorch_lightning.utilities import rank_zero_only

from load_data import SparseDataset
from models.superpoint import SuperPoint
from models.superglue import SuperGlue, ROOT_PATH
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error, hits_at_one,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified, masked_softmax)


class SuperGlueLightning(LightningModule):

    def __init__(self, args: DictConfig):
        super().__init__()
        self.hparams = args  # Will be logged to mlflow

        # make sure the flags are properly used
        assert not (args.exp.opencv_display and not args.exp.viz), 'Must use --viz with --opencv_display'
        assert not (args.exp.opencv_display and not args.exp.fast_viz), 'Cannot use --opencv_display without --fast_viz'
        assert not (args.exp.fast_viz and not args.exp.viz), 'Must use --viz with --fast_viz'
        assert not (args.exp.fast_viz and args.exp.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

        # store viz results
        # eval_output_dir = Path(f'{ROOT_PATH}/' + args.data.eval_output_dir)
        # eval_output_dir.mkdir(exist_ok=True, parents=True)
        # print('Will write visualization images to directory \"{}\"'.format(eval_output_dir))

        self.superglue = SuperGlue(args.model.superglue)
        self.superpoint = SuperPoint(args.model.superpoint)
        self.lr = None

    def prepare_data(self) -> None:
        self.train_set = SparseDataset(f'{ROOT_PATH}/' + self.hparams.data.train_path,
                                       self.hparams.model.superpoint.max_keypoints,
                                       self.hparams.data.resize,
                                       self.hparams.data.resize_float,
                                       self.hparams.model.superglue.min_keypoints,
                                       self.superpoint)
        self.val_set = SparseDataset(f'{ROOT_PATH}/' + self.hparams.data.val_path,
                                     self.hparams.model.superpoint.max_keypoints,
                                     self.hparams.data.resize,
                                     self.hparams.data.resize_float,
                                     self.hparams.model.superglue.min_keypoints,
                                     self.superpoint)
        self.train_set.files = self.train_set.files[:self.hparams.data.train_size]
        self.val_set.files = self.val_set.files[:self.hparams.data.val_size]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.train_set, shuffle=True, batch_size=self.hparams.data.batch_size.train,
                                           drop_last=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.val_set, shuffle=True, batch_size=self.hparams.data.batch_size.val,
                                           drop_last=True)

    def configure_optimizers(self):
        if self.lr is not None:
            self.hparams.optimizer.learning_rate = self.lr
        optimizer = torch.optim.Adam(self.superglue.parameters(), lr=self.hparams.optimizer.learning_rate)
        return optimizer

    def training_step(self, batch, batch_ind):

        for k in batch:
            if k != 'file_name' and k != 'image0' and k != 'image1':
                if type(batch[k]) == torch.Tensor:
                    batch[k] = Variable(batch[k])
                else:
                    batch[k] = Variable(torch.stack(batch[k]))

        # print(batch['keypoints0'].shape)
        data = self.superglue(batch)
        batch = {**batch, **data}
        # print(batch['keypoints0'].shape)

        if batch['skip_train']:  # image has no keypoint
            batch['loss'] = torch.zeros(1, requires_grad=True).type_as(batch['scores0'])
            return batch['loss']

        # Loss & Metrics
        self.log('train_hits_1', hits_at_one(batch).mean(), on_epoch=False, on_step=True, sync_dist=True)
        self.log('train_loss_m', batch['loss_m'], on_epoch=False, on_step=True, sync_dist=True)
        self.log('train_loss_um', batch['loss_um'], on_epoch=False, on_step=True, sync_dist=True)
        self.log('train_loss', batch['loss'], on_epoch=False, on_step=True, sync_dist=True)
        self.log('bin_score', self.superglue.bin_score, on_epoch=False, on_step=True, sync_dist=True)
        return batch['loss']

    def validation_step(self, batch, batch_ind):

        for k in batch:
            if k != 'file_name' and k != 'image0' and k != 'image1':
                if type(batch[k]) != torch.Tensor:
                    batch[k] = torch.stack(batch[k])

        data = self.superglue(batch)
        batch = {**batch, **data}

        if batch['skip_train']:  # image has no keypoint
            batch['loss'] = torch.zeros(1).type_as(batch['scores0'])
            return batch['loss']

        # Loss & Metrics
        self.log('val_hits_1', hits_at_one(batch).mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss', batch['loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_m', batch['loss_m'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_um', batch['loss_um'], on_epoch=True, on_step=False, sync_dist=True)
        return batch['loss']

    @rank_zero_only
    def on_train_epoch_end(self, outputs) -> None:
        for eval_data in self.hparams.eval:

            with open(f'{ROOT_PATH}/' + eval_data.pairs_list, 'r') as f:
                pairs = [l.split() for l in f.readlines()]

            if eval_data.max_length > -1:
                pairs = pairs[0:np.min([len(pairs), eval_data.max_length])]

            if eval_data.shuffle:
                random.Random(0).shuffle(pairs)

            if not all([len(p) == 38 for p in pairs]):
                raise ValueError(
                    'All pairs should have ground truth info for evaluation.'
                    'File \"{}\" needs 38 valid entries per row'.format(eval_data.pairs_list))

            # Load the SuperPoint and SuperGlue models.
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('Running inference on device \"{}\"'.format(device))
            config = {
                'superpoint': {
                    'nms_radius': eval_data.nms_radius,
                    'keypoint_threshold': eval_data.keypoint_threshold,
                    'max_keypoints': eval_data.max_keypoints
                },
                'superglue': self.hparams.model.superglue,
            }
            matching = Matching(config).eval().to(device)
            matching.superglue.load_state_dict(self.superglue.state_dict())

            # Create the output directories if they do not exist already.
            data_dir = Path(f'{ROOT_PATH}/' + eval_data.data_dir)
            # moving_dir = Path(f'{ROOT_PATH}/' + 'data/ScanNet/test_subset')
            print('Looking for data in directory \"{}\"'.format(data_dir))
            results_dir = Path(os.getcwd() + '/' + eval_data.results_dir)
            results_dir.mkdir(exist_ok=True, parents=True)
            print('Will write matches to directory \"{}\"'.format(results_dir))

            timer = AverageTimer(newline=True)
            for i, pair in enumerate(pairs):
                name0, name1 = pair[:2]
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                matches_path = results_dir / '{}_{}_matches.npz'.format(stem0, stem1)
                eval_path = results_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
                viz_path = results_dir / '{}_{}_matches.{}'.format(stem0, stem1, self.hparams.exp.viz_extension)
                viz_eval_path = results_dir / \
                                '{}_{}_evaluation.{}'.format(stem0, stem1, self.hparams.exp.viz_extension)

                # Handle --cache logic.
                do_match = True
                do_eval = True
                do_viz = self.hparams.exp.viz
                do_viz_eval = self.hparams.exp.viz
                # if opt.cache:
                #     if matches_path.exists():
                #         try:
                #             results = np.load(matches_path)
                #         except:
                #             raise IOError('Cannot load matches .npz file: %s' %
                #                           matches_path)
                #
                #         kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                #         matches, conf = results['matches'], results['match_confidence']
                #         do_match = False
                #     if opt.eval and eval_path.exists():
                #         try:
                #             results = np.load(eval_path)
                #         except:
                #             raise IOError('Cannot load eval .npz file: %s' % eval_path)
                #         err_R, err_t = results['error_R'], results['error_t']
                #         precision = results['precision']
                #         matching_score = results['matching_score']
                #         num_correct = results['num_correct']
                #         epi_errs = results['epipolar_errors']
                #         do_eval = False
                #     if opt.viz and viz_path.exists():
                #         do_viz = False
                #     if opt.viz and opt.eval and viz_eval_path.exists():
                #         do_viz_eval = False
                #     timer.update('load_cache')

                if not (do_match or do_eval or do_viz or do_viz_eval):
                    timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
                    continue

                # If a rotation integer is provided (e.g. from EXIF data), use it:
                if len(pair) >= 5:
                    rot0, rot1 = int(pair[2]), int(pair[3])
                else:
                    rot0, rot1 = 0, 0

                # Load the image pair.
                image0, inp0, scales0 = read_image(data_dir / name0, eval_data.resize, rot0, eval_data.resize_float)
                image1, inp1, scales1 = read_image(data_dir / name1, eval_data.resize, rot1, eval_data.resize_float)

                # Moving
                # os.makedirs(os.path.dirname(moving_dir / name0), exist_ok=True)
                # os.makedirs(os.path.dirname(moving_dir / name1), exist_ok=True)
                # shutil.copy(data_dir / name0, moving_dir / name0)
                # shutil.copy(data_dir / name1, moving_dir / name1)

                if image0 is None or image1 is None:
                    print('Problem reading image pair: {} {}'.format(data_dir / name0, data_dir / name1))
                    exit(1)
                timer.update('load_image')

                if do_match:
                    # Perform the matching.
                    with torch.no_grad():
                        pred = matching({'image0': inp0.cuda(), 'image1': inp1.cuda()})
                    pred_np = {}
                    for (k, v) in pred.items():
                        if isinstance(v, list):
                            pred_np[k] = v[0].cpu().numpy()
                        elif isinstance(v, torch.Tensor):
                            pred_np[k] = v[0].cpu().numpy()
                    pred = pred_np
                    # pred = {k: v[0].cpu().numpy() for k, v in pred.items() if isinstance(v, torch.Tensor)}
                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                    matches, conf = pred['matches0'], pred['matching_scores0']
                    timer.update('matcher')

                    # Write the matches to disk.
                    out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                                   'matches': matches, 'match_confidence': conf}
                    np.savez(str(matches_path), **out_matches)

                # Keep the matching keypoints.
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                if do_eval:
                    # Estimate the pose and compute the pose error.
                    assert len(pair) == 38, 'Pair does not have ground truth info'
                    K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
                    K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
                    T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

                    # Scale the intrinsics to resized image.
                    K0 = scale_intrinsics(K0, scales0)
                    K1 = scale_intrinsics(K1, scales1)

                    # Update the intrinsics + extrinsics if EXIF rotation was found.
                    if rot0 != 0 or rot1 != 0:
                        cam0_T_w = np.eye(4)
                        cam1_T_w = T_0to1
                        if rot0 != 0:
                            K0 = rotate_intrinsics(K0, image0.shape, rot0)
                            cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                        if rot1 != 0:
                            K1 = rotate_intrinsics(K1, image1.shape, rot1)
                            cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                        cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                        T_0to1 = cam1_T_cam0

                    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
                    correct = epi_errs < 5e-4
                    num_correct = np.sum(correct)
                    precision = np.mean(correct) if len(correct) > 0 else 0
                    matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

                    thresh = 1.  # In pixels relative to resized image size.
                    ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
                    if ret is None:
                        err_t, err_R = np.inf, np.inf
                    else:
                        R, t, inliers = ret
                        err_t, err_R = compute_pose_error(T_0to1, R, t)

                    # Write the evaluation results to disk.
                    out_eval = {'error_t': err_t,
                                'error_R': err_R,
                                'precision': precision,
                                'matching_score': matching_score,
                                'num_correct': num_correct,
                                'epipolar_errors': epi_errs}
                    np.savez(str(eval_path), **out_eval)
                    timer.update('eval')

                # if do_viz:
                #     # Visualize the matches.
                #     color = cm.jet(mconf)
                #     text = [
                #         'SuperGlue',
                #         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                #         'Matches: {}'.format(len(mkpts0)),
                #     ]
                #     if rot0 != 0 or rot1 != 0:
                #         text.append('Rotation: {}:{}'.format(rot0, rot1))
                #
                #     make_matching_plot(
                #         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                #         text, viz_path, stem0, stem1, opt.show_keypoints,
                #         opt.fast_viz, opt.opencv_display, 'Matches')
                #
                #     timer.update('viz_match')
                #
                # if do_viz_eval:
                #     # Visualize the evaluation results for the image pair.
                #     color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
                #     color = error_colormap(1 - color)
                #     deg, delta = ' deg', 'Delta '
                #     if not opt.fast_viz:
                #         deg, delta = '°', '$\\Delta$'
                #     e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
                #     e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
                #     text = [
                #         'SuperGlue',
                #         '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                #         'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
                #     ]
                #     if rot0 != 0 or rot1 != 0:
                #         text.append('Rotation: {}:{}'.format(rot0, rot1))
                #
                #     make_matching_plot(
                #         image0, image1, kpts0, kpts1, mkpts0,
                #         mkpts1, color, text, viz_eval_path,
                #         stem0, stem1, opt.show_keypoints,
                #         opt.fast_viz, opt.opencv_display, 'Relative Pose')
                #
                #     timer.update('viz_eval')

                timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

            # Collate the results into a final table and print to terminal.
            pose_errors = []
            precisions = []
            matching_scores = []
            for pair in pairs:
                name0, name1 = pair[:2]
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                eval_path = results_dir / \
                            '{}_{}_evaluation.npz'.format(stem0, stem1)
                results = np.load(eval_path)
                pose_error = np.maximum(results['error_t'], results['error_R'])
                pose_errors.append(pose_error)
                precisions.append(results['precision'])
                matching_scores.append(results['matching_score'])
            thresholds = [5, 10, 20]
            aucs = pose_auc(pose_errors, thresholds)
            aucs = [100. * yy for yy in aucs]
            prec = 100. * np.mean(precisions)
            ms = 100. * np.mean(matching_scores)
            print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
            print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
            print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2], prec, ms))

            self.log(f'{eval_data.name}/AUC_5', aucs[0], on_epoch=True, on_step=False)
            self.log(f'{eval_data.name}/AUC_10', aucs[1], on_epoch=True, on_step=False)
            self.log(f'{eval_data.name}/AUC_20', aucs[2], on_epoch=True, on_step=False)
            self.log(f'{eval_data.name}/Prec', prec, on_epoch=True, on_step=False)
            self.log(f'{eval_data.name}/MScore', ms, on_epoch=True, on_step=False)



    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
    #     if (self.trainer.global_step + 1) % 50 == 0 and len(outputs[0]) > 0:
    #         pass
        #     outputs = outputs[0][0]['extra']
        #     image0, image1 = outputs['image0'].cpu().numpy()[0] * 255., outputs['image1'].cpu().numpy()[0] * 255.
        #     kpts0, kpts1 = outputs['keypoints0'][0].cpu().numpy(), outputs['keypoints1'][0].cpu().numpy()
        #     matches, conf = outputs['matches0'].cpu().detach().numpy(), outputs['matching_scores0'].cpu().detach().numpy()
        #     image0 = read_image_modified(image0, self.hparams.data.resize, self.hparams.data.resize_float)
        #     image1 = read_image_modified(image1, self.hparams.data.resize, self.hparams.data.resize_float)
        #     valid = matches > -1
        #     mkpts0 = kpts0[valid]
        #     mkpts1 = kpts1[matches[valid]]
        #     mconf = conf[valid]
        #     viz_path = self.hparams.data.eval_output_dir + \
        #                f'{str(self.trainer.global_step)}_matches.{self.hparams.exp.viz_extension}'
        #     color = cm.jet(mconf)
        #     stem = outputs['file_name']
        #     text = []
        #
        #     make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, viz_path, stem, stem,
        #                        self.hparams.exp.show_keypoints, self.hparams.exp.fast_viz, self.hparams.exp.opencv_display,
        #                        'Matches')