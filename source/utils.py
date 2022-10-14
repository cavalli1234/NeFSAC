from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import numba as nb
import pycolmap as pyc
import torch
import cv2

def resize_images(im1, im2, target_width, p1, p2):
    w1 = w2 = target_width
    h1, h2 = int(im1.shape[0] / im1.shape[1] * w1), int(im2.shape[0] / im2.shape[1] * w2)
    p1 = p1 * np.array([w1 / im1.shape[1], h1 / im1.shape[0]])
    p2 = p2 * np.array([w2 / im2.shape[1], h2 / im2.shape[0]])
    return cv2.resize(im1, (w1, h1)), cv2.resize(im2, (w2, h2)), p1, p2

def plot_matches(im1, im2, p1, p2, s, ax=None):
    im1 = np.asarray(im1)/255.
    im2 = np.asarray(im2)/255.
    h1, w1 = im1.shape[0], im1.shape[1]
    h2, w2 = im2.shape[0], im2.shape[1]
    p1r, p2r = np.copy(p1), np.copy(p2)
    h1, w1 = im1.shape[0], im1.shape[1]
    if ax is None:
        ax = plt.subplot()
    im1, im2, p1r, p2r = resize_images(im1, im2, 512, p1r, p2r)
    stitched_rgb = np.concatenate([im1, im2], axis=0)
    ax.imshow(stitched_rgb)
    off = np.array([0, im1.shape[0]])
    green = np.array([0., 1., 0.])
    for xy1, xy2, xy1r, xy2r in zip(p1, p2, p1r, p2r):
        col = green
        ax.add_patch(plt.Circle(xy1r, 2, color=col))
        ax.add_patch(plt.Circle(xy2r + off, 2, color=col))
        ax.plot(*zip(xy1r, xy2r + off), color=col, linewidth=0.2)
    for i, s1 in enumerate(s):
        ax.text(20, 20 + 20 * i, s1, fontsize='small', weight='bold', fontstyle='italic', color=(1., 1., 1.))

def read_and_match(im1_path, im2_path):
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    k1, k2 = match_images(im1, im2)
    return torch.tensor(np.concatenate([k1, k2], axis=-1), dtype=torch.float32), np.asarray(im1), np.asarray(im2)

def extract_sift(im, maxkp=8000, num_octaves=4, octave_resolution=3, first_octave=-1, peak_thresh=2/300, edge_thresh=10):
    if isinstance(im, str):
        with open(im, 'rb') as imf:
            img = Image.open(imf).convert('L')
    elif isinstance(im, Image.Image):
        im = ImageOps.grayscale(im)
    else:
        raise TypeError
    img = np.array(im).astype(np.float32) / 255.

    kp, scores, desc = pyc.extract_sift(img, num_octaves=num_octaves,
                                        octave_resolution=octave_resolution,
                                        first_octave=first_octave,
                                        peak_thresh=peak_thresh,
                                        edge_thresh=edge_thresh)
    if len(kp) > maxkp:
        indices = np.argpartition(-scores, maxkp)[:maxkp]
        kp = kp[indices]
        desc = desc[indices]
    kp = kp[:, :2]
    return kp, desc


def ratio_test_nn_match(d1, d2, ratio, do_mnn=False, sqrdist_mat=None):
    if sqrdist_mat is None:
        sqrdist_mat = 1. - d1 @ d2.T

    nn12, snn12 = np.argpartition(sqrdist_mat, (1, 2) if sqrdist_mat.shape[1] > 2 else (1,), axis=1)[:, :2].T
    nn21, snn21 = np.argpartition(sqrdist_mat, (1, 2) if sqrdist_mat.shape[0] > 2 else (1,), axis=0)[:2, :]

    range12 = np.arange(len(nn12))
    sqrratios = sqrdist_mat[range12, nn12]/sqrdist_mat[range12, snn12]
    if do_mnn:
        mnn = nn21[nn12] == range12
        corrs = (sqrratios < (ratio ** 2)) & mnn
    else:
        corrs = sqrratios < (ratio ** 2)

    idx1 = np.argwhere(corrs).flatten()
    idx2 = nn12[corrs]
    return idx1, idx2

def list_image_couples(n_frames, interval=1):
    out = set()
    for i in range(n_frames):
        if i+interval < n_frames:
            out.add((i, i+interval))
    return out

def match_images(i1, i2):
    kp1, desc1 = extract_sift(i1)
    kp2, desc2 = extract_sift(i2)
    idx1, idx2 = ratio_test_nn_match(desc1, desc2, 0.8)
    return kp1[idx1], kp2[idx2]

def rt_error(R_gt, T_gt, R_est, T_est):
    if not isinstance(R_est, torch.Tensor):
        R_est = torch.tensor(np.array(R_est))
    if not isinstance(T_est, torch.Tensor):
        T_est = torch.tensor(np.array(T_est))
    if not isinstance(R_gt, torch.Tensor):
        R_gt = torch.tensor(np.array(R_gt))
    if not isinstance(T_gt, torch.Tensor):
        T_gt = torch.tensor(np.array(T_gt))
    assert R_est.ndim == T_est.ndim + 1
    assert R_gt.ndim == T_gt.ndim + 1
    if R_est.ndim == 2:
        R_est = R_est.unsqueeze(0)
        T_est = T_est.unsqueeze(0)
    if R_gt.ndim == 2:
        R_gt = R_gt.unsqueeze(0)
        T_gt = T_gt.unsqueeze(0)
    assert R_est.ndim == 3
    assert R_gt.ndim == 3
    R_est, T_est, R_gt, T_gt = map(lambda t: t.to(torch.double), (R_est, T_est, R_gt, T_gt))
    # P2 = R2 @ (R1T @ ( x - t1) + t2 = R2 @ R1T @ x - R2 @ R1T @ t1 + t2
    # P2 = R @ (x - t) = Rx - Rt
    T_gt = T_gt.div(T_gt.norm(dim=-1, keepdim=True))
    T_est = T_est.div(T_est.norm(dim=-1, keepdim=True))

    r_err = (torch.diagonal(R_est @ R_gt.transpose(-1, -2), dim1=-1, dim2=-2).sum(dim=-1)-1).div(2.).clamp(-1., 1.).acos() / np.pi * 180
    t_err = (T_gt * T_est).sum(dim=-1).clamp(-1., 1.).acos() / np.pi * 180

    return r_err, t_err


@nb.njit
def sampson_distance(source_point, destination_point, model):
    return np.sqrt(squared_sampson_distance(source_point, destination_point, model))

@nb.njit
def squared_sampson_distance(source_point, destination_point, model):
    x1 = source_point[0]
    y1 = source_point[1]
    x2 = destination_point[0]
    y2 = destination_point[1]

    e11 = model[0, 0]
    e12 = model[0, 1]
    e13 = model[0, 2]
    e21 = model[1, 0]
    e22 = model[1, 1]
    e23 = model[1, 2]
    e31 = model[2, 0]
    e32 = model[2, 1]
    e33 = model[2, 2]

    rxc = e11 * x2 + e21 * y2 + e31
    ryc = e12 * x2 + e22 * y2 + e32
    rwc = e13 * x2 + e23 * y2 + e33
    r = (x1 * rxc + y1 * ryc + rwc)
    rx = e11 * x1 + e12 * y1 + e13
    ry = e21 * x1 + e22 * y1 + e23

    return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry)

@nb.njit
def cross_mat(a):
    assert a.shape[-1] == 3
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    out = np.zeros(a.shape + (3,), a.dtype)
    out[..., 0, 1] = -a3
    out[..., 0, 2] = a2
    out[..., 1, 2] = -a1
    out[..., 1, 0] = a3
    out[..., 2, 0] = -a2
    out[..., 2, 1] = a1
    return out

@nb.njit
def compute_F(R, T, K1inv, K2inv):
    F = (K2inv.T @ cross_mat(T/np.linalg.norm(T)) @ R @ K1inv)
    return F
