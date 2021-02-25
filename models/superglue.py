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
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

ROOT_PATH = dirname(dirname(abspath(__file__)))


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = torch.cat(inputs, dim=1)
        return self.encoder(inputs)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
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

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
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

        self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        if self.config['weights'] is not None:
            path = Path(__file__).parent.parent
            path = path / self.config['weights']

            state_dict = {k.replace('superglue.', ''): v for k, v in torch.load(path)['state_dict'].items()
                          if 'superpoint' not in k}
            self.load_state_dict(state_dict)
            # self.load_state_dict(SuperGlueLightning.load_from_checkpoint(path).superpoint.state_dict())
            print('Loaded SuperGlue model (\"{}\" weights)'.format(self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'].float(), data['descriptors1'].float()
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()
    
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }
        
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'].float())
        desc1 = desc1 + self.kenc(kpts1, data['scores1'].float())

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly
        if 'all_matches' in data:
            loss = []
            all_matches = data['all_matches']  # shape=torch.Size([1, 87, 2])
            for i in range(len(all_matches[0])):
                x = all_matches[0][i][0]
                y = all_matches[0][i][1]
                loss.append(-torch.log( scores[0][x][y].exp() )) # check batch size == 1 ?

            # for p0 in unmatched0:
            #     loss += -torch.log(scores[0][p0][-1])
            # for p1 in unmatched1:
            #     loss += -torch.log(scores[0][-1][p1])
            loss_mean = torch.mean(torch.stack(loss))
            loss_mean = torch.reshape(loss_mean, (1, -1))
            # scores big value or small value means confidence? log can't take neg value
            return {
                'matches0': indices0[0], # use -1 for invalid match
                'matches1': indices1[0], # use -1 for invalid match
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0],
                'loss': loss_mean[0],
                'skip_train': False
            }
        else:
            return {
                'matches0': indices0[0],  # use -1 for invalid match
                'matches1': indices1[0],  # use -1 for invalid match
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0],
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

    def prepare_data(self) -> None:
        self.train_set = SparseDataset(f'{ROOT_PATH}/' + self.hparams.data.train_path,
                                       self.hparams.model.superpoint.max_keypoints,
                                       self.hparams.data.resize,
                                       self.hparams.data.resize_float,
                                       self.superpoint)
        self.val_set = SparseDataset(f'{ROOT_PATH}/' + self.hparams.data.val_path,
                                     self.hparams.model.superpoint.max_keypoints,
                                     self.hparams.data.resize,
                                     self.hparams.data.resize_float,
                                     self.superpoint)
        self.val_set.files = self.val_set.files[:self.hparams.data.val_size]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.train_set, shuffle=False, batch_size=self.hparams.data.batch_size,
                                           drop_last=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.val_set, shuffle=False, batch_size=self.hparams.data.batch_size,
                                           drop_last=True)

    def configure_optimizers(self):
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
        for k, v in batch.items():
            batch[k] = v[0]

        batch = {**batch, **data}

        if batch['skip_train']:  # image has no keypoint
            return None

        # process loss
        self.log('train_loss', batch['loss'], on_epoch=False, on_step=True, sync_dist=True)
        return batch

    def validation_step(self, batch, batch_ind):

        for k in batch:
            if k != 'file_name' and k != 'image0' and k != 'image1':
                if type(batch[k]) != torch.Tensor:
                    batch[k] = torch.stack(batch[k])

        data = self.superglue(batch)
        for k, v in batch.items():
            batch[k] = v[0]

        batch = {**batch, **data}

        if batch['skip_train']:  # image has no keypoint
            return None

        # process loss
        self.log('val_loss', batch['loss'], on_epoch=True, on_step=False, sync_dist=True)
        return batch

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

