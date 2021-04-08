# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from torch.autograd import Variable
import matplotlib.cm as cm
from os.path import abspath, dirname

from load_data import SparseDataset
from models.superpoint import SuperPoint
from models.masked_bn import MaskedBatchNorm1d
from models.utils import (compute_pose_error, compute_epipolar_error, hits_at_one,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified, masked_softmax)

ROOT_PATH = dirname(dirname(abspath(__file__)))


class MLP(nn.Module):
    def __init__(self, channels: list, do_bn=True):
        """ Multi-layer perceptron """
        super().__init__()
        n = len(channels)
        layers = []
        for i in range(1, n):
            layers.append(
                nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
            if i < (n - 1):
                if do_bn:
                    layers.append(MaskedBatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers).float()

    def forward(self, input, input_mask=None):
        x = input
        for layer in self.layers:
            if isinstance(layer, MaskedBatchNorm1d):
                x = layer(x, input_mask=input_mask).float()
            else:
                x = layer(x)
        return x


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder.layers[-1].bias, 0.0)

    def forward(self, kpts, scores, mask):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = torch.cat(inputs, dim=1)
        return self.encoder(inputs, mask.unsqueeze(1))


def attention(query, key, value, query_mask, key_value_mask):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5

    # key_value_mask = key_value_mask.clone().float()
    # query_mask = query_mask.clone().float()
    # key_value_mask = key_value_mask.masked_fill(key_value_mask == 0.0, -float('Inf'))
    # query_mask = query_mask.masked_fill(query_mask == 0.0, -float('Inf'))

    # * scores
    # scores = scores.masked_fill(torch.isinf(scores), -float('Inf'))

    prob = masked_softmax(scores, mask=key_value_mask.unsqueeze(1).unsqueeze(2) * query_mask.unsqueeze(1).unsqueeze(3), dim=-1)
    # prob[prob.isnan()] = -10**6
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, query_mask, key_value_mask):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, query_mask, key_value_mask)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp.layers[-1].bias, 0.0)

    def forward(self, x, source, x_mask, source_mask):
        message = self.attn(x, source, source, x_mask, source_mask)
        return self.mlp(torch.cat([x, message], dim=1), x_mask.unsqueeze(1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, mask0, mask1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1, src0_mask, src1_mask = desc1, desc0, mask1, mask0
            else:  # if name == 'self':
                src0, src1, src0_mask, src1_mask = desc0, desc1, mask0, mask1
            delta0, delta1 = layer(desc0, src0, mask0, src0_mask), layer(desc1, src1, mask1, src1_mask)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        u = u.masked_fill(u.isinf() | u.isnan(), -float('Inf'))
        # assert u.isnan().any()
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        v = v.masked_fill(v.isinf() | v.isnan(), -float('Inf'))
        # assert v.isnan().any()
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int, mask_rows, mask_cols):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    # b, m, n = scores.shape[0], mask_rows.sum(1), mask_cols.sum(1)

    scores = (mask_cols.unsqueeze(1) * mask_rows.unsqueeze(2) + 1e-45).log() + scores

    one = scores.new_tensor(1)
    # ms, ns = (m * one).to(scores), (n * one).to(scores)
    ms, ns = (mask_rows.sum(1) * one).to(scores), (mask_cols.sum(1) * one).to(scores)

    bins0 = torch.cat([torch.cat(
        [alpha.expand(1, ms[i].int(), 1), torch.tensor(-float('Inf')).type_as(alpha).expand(1, (m - ms[i]).int(), 1)], 1) for i in
                       range(ms.shape[0])], 0)
    # bins0[mask_rows == 0.0] = torch.tensor(-float('Inf')).type_as(scores)
    bins1 = torch.cat([torch.cat(
        [alpha.expand(1, 1, ns[i].int()), torch.tensor(-float('Inf')).type_as(alpha).expand(1, 1, (n - ns[i]).int())], 2) for i in
                       range(ns.shape[0])], 0)
    # bins1[:, mask_cols == 0.0] = torch.tensor(-float('Inf')).type_as(scores)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    # log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_mu = torch.cat([norm.repeat(m, 1), (ns.log() + norm)[None]]).T
    log_mu[:, :-1][mask_rows == 0.0] = torch.tensor(-float('Inf')).type_as(scores)
    # log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_nu = torch.cat([norm.repeat(n, 1), (ms.log() + norm)[None]]).T
    # log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    log_nu[:, :-1][mask_cols == 0.0] = torch.tensor(-float('Inf')).type_as(scores)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm.unsqueeze(-1).unsqueeze(-1)  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,  # 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],  # ,  256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc0 = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])
        # self.kenc1 = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        if self.config['weights'] is not None:
            path = Path(__file__).parent.parent
            path = path / self.config['weights']

            model = torch.load(path, map_location=torch.device('cpu'))
            if 'state_dict' in model:
                state_dict = {k.replace('superglue.', ''): v for k, v in model['state_dict'].items() if 'superpoint' not in k}
            else:
                state_dict = model
            self.load_state_dict(state_dict)
            # self.load_state_dict(SuperGlueLightning.load_from_checkpoint(path).superpoint.state_dict())
            print('Loaded SuperGlue model (\"{}\" weights)'.format(self.config['weights']))

    def _discard_empty(self, data):
        mask = data['mask0']
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v[mask.sum(1) != 0.0]
            else:
                # print(len(v), len(mask.sum(1)))
                pass
                # data[k] = [vv for (i, vv) in enumerate(v) if (mask.sum(1)[i] != 0.0)]
        return data


    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        data = self._discard_empty(data)

        desc0, desc1 = data['descriptors0'].float(), data['descriptors1'].float()
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()
        mask0, mask1 = data['mask0'], data['mask1']

        if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:  # no keypoints
            # shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                # 'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                # 'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                # 'matching_scores0': kpts0.new_zeros(shape0)[0],
                # 'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc0(kpts0, data['scores0'].float(), mask0)
        desc1 = desc1 + self.kenc0(kpts1, data['scores1'].float(), mask1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1, mask0, mask1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        log_scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'],
                                           mask_rows=mask0, mask_cols=mask1)

        torch.testing.assert_allclose(mask0.sum(1) + mask1.sum(1),
                                      log_scores.exp().masked_fill(log_scores.exp().isnan(), 0.0).sum((1, 2)))

        # Get the matches with score above "match_threshold".
        max0, max1 = log_scores[:, :-1, :-1].max(2), log_scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = log_scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly
        if 'all_matches' in data:
            all_matches = data['all_matches']

            # True Matches score
            loss_m = - log_scores[:, all_matches[:, :, 0]].diagonal(dim1=0, dim2=1).transpose(0, 2)
            loss_m = loss_m[:, all_matches[:, :, 1]].diagonal(dim1=0, dim2=1).diagonal(dim1=0, dim2=1)
            loss_m *= data['all_matches_mask']
            # lossv2 = lossv2.sum(1)
            n = data['all_matches_mask'].sum(1)
            loss_m = loss_m.sum(1) #/ n.masked_fill(n == 0.0, 1.0)

            # Dustbin loss for unmatched points
            non_ind_rows = [[ii for ii in range(mask0[b].sum().int()) if ii not in b_ix[data['all_matches_mask'][b].bool()]] for (b, b_ix) in
                            enumerate(all_matches[:, :, 0])]
            non_ind_cols = [[ii for ii in range(mask1[b].sum().int()) if ii not in b_ix[data['all_matches_mask'][b].bool()]] for (b, b_ix) in
                            enumerate(all_matches[:, :, 1])]
            nr = (mask0.sum(1) - data['all_matches_mask'].sum(1))
            nc = (mask1.sum(1) - data['all_matches_mask'].sum(1))
            loss_um = - torch.stack([log_scores[b, non_ind_rows[b], -1].sum() for b in range(log_scores.shape[0])])
                      # / nr.masked_fill(nr == 0.0, 1.0)
            loss_um -= torch.stack([log_scores[b, -1, non_ind_cols[b]].sum() for b in range(log_scores.shape[0])])
                      # / nc.masked_fill(nc == 0.0, 1.0)

            # Masking empty images
            assert not loss_m.isnan().any()
            assert not loss_um.isnan().any()

            return {
                'matches0': indices0,  # use -1 for invalid match
                'matches1': indices1,  # use -1 for invalid match
                'matching_scores0': mscores0,
                'matching_scores1': mscores1,
                'loss_m': loss_m.mean(),
                'loss_um': loss_um.mean(),
                'loss': loss_m.mean() + loss_um.mean(),
                'skip_train': False
            }
        else:
            return {
                'matches0': indices0,  # use -1 for invalid match
                'matches1': indices1,  # use -1 for invalid match
                'matching_scores0': mscores0,
                'matching_scores1': mscores1,
                'loss_m': None,
                'loss_um': None,
                'loss': None,
                'skip_train': True
            }


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
        eval_output_dir = Path(f'{ROOT_PATH}/' + args.data.eval_output_dir)
        eval_output_dir.mkdir(exist_ok=True, parents=True)
        print('Will write visualization images to directory \"{}\"'.format(eval_output_dir))

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

        data = self.superglue(batch)
        batch = {**batch, **data}

        if batch['skip_train']:  # image has no keypoint
            return None

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
            return None

        # Loss & Metrics
        self.log('val_hits_1', hits_at_one(batch).mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss', batch['loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_m', batch['loss_m'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_loss_um', batch['loss_um'], on_epoch=True, on_step=False, sync_dist=True)
        return batch['loss']

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        pass
        # if (self.trainer.global_step + 1) % 50 == 0 and len(outputs[0]) > 0:
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
