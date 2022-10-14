import torch
import numpy as np
import random
import cv2
from copy import deepcopy
from pathlib import Path
import pykitti as pyk
from .utils import match_images, list_image_couples, rt_error, compute_F, sampson_distance
from scipy.spatial.transform import Rotation
import shelve
from collections.abc import Iterable


class CompoundDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, name=None):
        super().__init__()
        self.datasets = datasets
        self.name = name
        self.iters_done = 0

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        self.iters_done += 1
        iternum = self.iters_done
        iters = [iter(d) for d in self.datasets]
        nm = 0
        while len(iters) > 0:
            i = np.random.randint(len(iters))
            it = iters[i]
            try:
                yield next(it)
                nm += 1
            except StopIteration:
                iters.remove(it)


class KittiSequenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, sequence_num, bsize=256, interval=1, sample_size=5, maxlen=None, analytical='RT', kitti_base_path=None, data_cache_path=None):
        if data_cache_path is not None:
            self.cache = shelve.open((Path(cache_path) / f"/sq{sequence_num}_int{interval}").as_posix())
            self.caching = True
            print(f"Opening cache with {len(self.cache)} pre-existing entries")
        else:
            print("No location specified for the data cache persistency. The data for this run will not be persisted.")
            self.cache = dict()
            self.caching = False
        self.kitti_reader = pyk.odometry(kitti_base_path, f"{sequence_num:02d}")
        self.image_indices = list(list_image_couples(len(self.kitti_reader), interval))
        self.K = self.kitti_reader.calib.K_cam0
        self.Kinv = np.linalg.inv(self.K)
        self.sample_size = sample_size
        self.bsize = bsize
        self.ll = maxlen or len(self.image_indices)
        assert sample_size in (5, 7)
        self.F_mode = sample_size == 7
        self.analytical = analytical


    def __new__(cls, sequence_num, bsize=None, interval=None, *args, **kwargs):
        if isinstance(sequence_num, Iterable) and isinstance(interval, Iterable):
            return CompoundDataset([KittiSequenceDataset(sq, bsize, i, *args, **kwargs) for sq in sequence_num for i in interval])
        elif isinstance(sequence_num, Iterable):
            return CompoundDataset([KittiSequenceDataset(sq, bsize, interval, *args, **kwargs) for sq in sequence_num])
        elif isinstance(interval, Iterable):
            return CompoundDataset([KittiSequenceDataset(sequence_num, bsize, i, *args, **kwargs) for i in interval])
        return super(KittiSequenceDataset, cls).__new__(cls)


    def __len__(self):
        return self.ll

    def __iter__(self):
        ii = deepcopy(self.image_indices)
        random.shuffle(ii)
        ii = ii[:self.ll]
        for i1, i2 in ii:
            dictkey = f"{i1}_{i2}"
            rt = np.linalg.inv(self.kitti_reader.poses[i2]) @ self.kitti_reader.poses[i1]
            if dictkey in self.cache.keys():
                k1, k2 = self.cache[dictkey]
            else:
                im1 = self.kitti_reader.get_cam0(i1)
                im2 = self.kitti_reader.get_cam0(i2)
                k1, k2 = match_images(im1, im2)
                self.cache[dictkey] = (k1, k2)
            minimal_samples = np.array([np.random.choice(len(k1), size=self.sample_size, replace=False) for _ in range(self.bsize)])
            batch = np.concatenate([k1[minimal_samples], k2[minimal_samples]], axis=-1)
            if self.F_mode:
                batch_unproj = batch / np.array([1241, 378, 1241, 378]) * 2 - 1.  # Assuming a standard image size of 1241 x 378 for KITTI
                batch_rt = batch
            else:
                batch_unproj = np.concatenate([self.unproj(k1[minimal_samples]), self.unproj(k2[minimal_samples])], axis=-1)
                batch_rt = batch_unproj
            rt_err = np.array([self.compute_rt_error(k, rt) for k in batch_rt])
            gtrt_err = rt_err[:, 0]
            anrt_err = rt_err[:, 1]
            err_sort = np.argsort(gtrt_err)
            ok_samples = np.concatenate([err_sort[:self.bsize//2], err_sort[-self.bsize//2:]])
            gtrt_err = gtrt_err[ok_samples]
            anrt_err = anrt_err[ok_samples]
            batch = batch[ok_samples]
            batch_unproj = batch_unproj[ok_samples]
            s_err = np.array([self.compute_samp_error(k, rt) for k in batch])
            assert batch.shape == (self.bsize, self.sample_size, 4)
            yield batch_unproj.astype(np.float32), gtrt_err.astype(np.float32), anrt_err.astype(np.float32), s_err.astype(np.float32)

    def unproj(self, k):
        return k @ self.Kinv[:2, :2] + self.Kinv[:2, 2]

    def analytical_model(self, pose):
        if self.analytical == 'RT':
            R, T = pose
            yxz_rot = Rotation.from_matrix(R).as_euler('YXZ')
            yxz_rot[0] = 0.
            rot_err = Rotation.from_euler('YXZ', yxz_rot).magnitude()
            t_err = np.pi/2 - np.arccos(np.abs(T[1]/np.linalg.norm(T)))
            return np.rad2deg(np.maximum(rot_err, t_err))
        else:
            return 0.

    def compute_rt_error(self, k12, gtrt):
        k1, k2 = k12[:, :2], k12[:, 2:]
        R_gt, T_gt = gtrt[:3, :3], gtrt[:3, 3]
        fmt = lambda k: np.pad(k.astype(np.double), ((0, 0), (0, 1)), constant_values=1)
        if self.F_mode:
            Fs, _ = cv2.findFundamentalMat(k1,
                                           k2,
                                           cv2.FM_7POINT)
            if Fs is None or len(Fs) == 0:
                poses = []
            else:
                Fs = np.reshape(Fs, (Fs.shape[0]//3, 3, 3))
                k2u = cv2.undistortPoints(k2[:, None], self.K, 0)
                k1u = cv2.undistortPoints(k1[:, None], self.K, 0)
                def F_to_pose(f):
                    E = self.K.T @ f @ self.K
                    _, R, t, _ = cv2.recoverPose(E, k1u, k2u)
                    return R, t[:, 0]
                poses = [F_to_pose(F) for F in Fs]

        else:
            # poselibposes = poselib.relpose_5pt(fmt(k1), fmt(k2))
            Es, _ = cv2.findEssentialMat(k1,
                                         k2)
            if Es is None or len(Es) == 0:
                poses = []
            else:
                Es = np.reshape(Es, (Es.shape[0]//3, 3, 3))
                def E_to_pose(e):
                    _, R, t, _ = cv2.recoverPose(e, k1, k2)
                    return R, t[:, 0]
                poses = [E_to_pose(E) for E in Es]
        if len(poses) > 0:
            R = np.array([pose[0] for pose in poses])
            T = np.array([pose[1] for pose in poses])
            errs = rt_error(R_gt, T_gt, R, T)
            err_gt = torch.maximum(*errs).min().item()
            err_an = min(self.analytical_model(pose) for pose in poses)
        else:
            err_gt = 180.
            err_an = 180.
        return err_gt, err_an

    def compute_samp_error(self, k12, gtrt):
        k1, k2 = k12[:, :2], k12[:, 2:]
        R_gt, T_gt = gtrt[:3, :3], gtrt[:3, 3]
        F = compute_F(R_gt, T_gt, self.Kinv, self.Kinv)
        se = [sampson_distance(p1, p2, F) for p1, p2 in zip(k1, k2)]
        return max(se)

    def __del__(self):
        if self.caching:
            self.cache.close()

