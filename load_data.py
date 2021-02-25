import numpy as np
import torch
import os
import cv2
import math
import datetime

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

from models.superpoint import SuperPoint
from models.utils import frame2tensor, process_resize

np.random.seed(42)


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path, nfeatures, resize, resize_float, detector='sift'):

        self.files = []
        self.files += [train_path + f for f in os.listdir(train_path)]

        self.nfeatures = nfeatures
        self.resize = resize
        self.resize_float = resize_float
        # detector = 'sift'
        if detector == 'sift':
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

            def detect(img):

                kp, descs = sift.detectAndCompute(img, None)
                kp_num = min(self.nfeatures, len(kp))
                kp = kp[:kp_num]
                if descs is not None:
                    descs = descs[:kp_num, :]

                kp_np = np.array([(k.pt[0], k.pt[1]) for k in kp])
                scores_np = np.array([k.response for k in kp])
                return kp_np, descs, scores_np

            self.detect = detect
        elif isinstance(detector, SuperPoint):
            def detect(img):
                with torch.no_grad():
                    result = detector({'image': frame2tensor(img.astype('float32'))})
                return result['keypoints'][0].cpu().float().numpy(), \
                       result['descriptors'][0].cpu().float().numpy(), \
                       result['scores'][0].cpu().float().numpy()
            self.detect = detect
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def _pad_and_mask(self, arr, axis, val=np.nan):
        if axis == 0:
            new_arr = np.pad(arr, pad_width=((0, self.nfeatures - arr.shape[0]), (0, 0)), mode='constant', constant_values=val)
            mask = np.array([1.0] * arr.shape[0] + [0.0] * (self.nfeatures - arr.shape[0]))
        else:
            new_arr = np.pad(arr, pad_width=((0, 0), (0, self.nfeatures - arr.shape[1])), mode='constant', constant_values=val)
            mask = np.array([1.0] * arr.shape[1] + [0.0] * (self.nfeatures - arr.shape[1]))
        return new_arr, mask


    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # Resize
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)

        if self.resize_float:
            image = cv2.resize(image.astype('float32'), (w_new, h_new))
        else:
            image = cv2.resize(image, (w_new, h_new)).astype('float32')



        width, height = image.shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

        kp1, kp2, all_matches = [], [], []

        # while len(kp1) < 2 or len(kp2) < 2 or len(all_matches) < 2:

        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        # get the corresponding warped image
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0])) # return an image type

        # extract keypoints of the image pair using SIFT
        kp1_np, descs1, scores1_np = self.detect(image)
        kp2_np, descs2, scores2_np = self.detect(warped)

        # skip this image pair if no keypoints detected in image
        if len(kp1_np) < 2 or len(kp2_np) < 2:
            return{
                'keypoints0': torch.zeros([0, 0, 2]),
                'keypoints1': torch.zeros([0, 0, 2]),
                'descriptors0': torch.zeros([0, 2]),
                'descriptors1': torch.zeros([0, 2]),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            }

        # obtain the matching matrix of the image pair
        # matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        # MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        # MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = MN.T  # np.concatenate([MN, MN2, MN3], axis=1).T
        # all_matches = np.unique(all_matches, axis=0)

        # if len(all_matches) < 2:
        #     continue

        # kp1_np = kp1_np.reshape((2, -1))
        # kp2_np = kp2_np.reshape((2, -1))
        # scores1_np = scores1_np.reshape((1, -1))
        # scores2_np = scores2_np.reshape((1, -1))
        # descs1 = np.pad((descs1 / 256.).T, pad_width=((0, 128), (0, 0)), constant_values=0.0)
        # descs2 = np.pad((descs2 / 256.).T, pad_width=((0, 128), (0, 0)), constant_values=0.0)

        image = torch.from_numpy(image/255.)[None].cuda().float()
        warped = torch.from_numpy(warped/255.)[None].cuda().float()


        # Padding
        kp1_np, mask1 = self._pad_and_mask(kp1_np, axis=0, val=np.nan)
        kp2_np, mask2 = self._pad_and_mask(kp2_np, axis=0, val=np.nan)
        descs1, _ = self._pad_and_mask(descs1, axis=1, val=np.nan)
        descs2, _ = self._pad_and_mask(descs2, axis=1, val=np.nan)
        scores1_np = np.pad(scores1_np, pad_width=((0, self.nfeatures - scores1_np.shape[0]), ), mode='constant',
                            constant_values=np.nan)
        scores2_np = np.pad(scores2_np, pad_width=((0, self.nfeatures - scores2_np.shape[0]),), mode='constant',
                            constant_values=np.nan)
        all_matches, all_matches_mask = self._pad_and_mask(all_matches, axis=0, val=0)

        return{
            'keypoints0': kp1_np,
            'keypoints1': kp2_np,
            'descriptors0': descs1,
            'descriptors1': descs2,
            'scores0': scores1_np,
            'scores1': scores2_np,
            'mask0': mask1,
            'mask2': mask2,
            'image0': image,
            'image1': warped,
            'all_matches': all_matches,
            'all_matches_mask': all_matches_mask,
            'file_name': file_name
        }

