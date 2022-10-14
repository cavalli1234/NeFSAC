from matplotlib import pyplot as plt
import numpy as np
import torch
import time
from source.utils import read_and_match, plot_matches
from source.ransac.fundamental_matrix_estimator import FundamentalMatrixEstimator
from source.ransac.msac_score import MSACScore
from source.ransac.samplers import UniformSampler, NeFSACSampler
from source.ransac.core import ransac

def main(im1_path, im2_path, dev='cuda'):
    # Fix the seed for reproducibility
    torch.manual_seed(0)

    # Initialize the solver, this is the 7-point solver for F matrix
    slv = FundamentalMatrixEstimator()

    # Initialize the scoring method, this uses MSAC scoring
    score = MSACScore(2.)

    print("Reading and matching images...")
    data, im1, im2 = read_and_match(im1_path, im2_path)
    print("Done.")
    data = data.to(dev)

    # Initialize the samplers
    sampler_u = UniformSampler(4096, 7)
    sampler_n = NeFSACSampler(4096, 7, 'pretrained_models/nefsac_F_phototourism.pt',
                              dev=dev, keep_rate=0.2, effective_iters_multiplier=1)
    # NeFSAC normalizes the input correspondences with the known image sizes for F estimation
    sampler_n.set_F_normalization(im1.shape, im2.shape)
    fig, axs = plt.subplots(1, 2)
    names = ["Uniform", "NeFSAC"]
    for i, sampler in enumerate([sampler_u, sampler_n]):
        # Warmup for timing
        inl = ransac(data, sampler, slv.estimate_model, score.select_inliers, eps=1., max_iters=1)
        print(f"Running RANSAC for baseline {names[i]}...")
        torch.cuda.synchronize()
        t1 = time.time()
        # Run the actual RANSAC
        inl = ransac(data, sampler, slv.estimate_model, score.select_inliers, eps=1e-30, max_iters=1e+9)
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"Baseline {names[i]} done.")
        # Plot the matches and timings
        inliers_np = data[inl].cpu().numpy().astype(np.double)
        p1, p2 = inliers_np[:, :2], inliers_np[:, 2:4]
        plot_matches(im1, im2, p1, p2, [names[i], f"Time: {t2-t1:.2f}s", f"Inlier rate found: {inl.float().mean().item() * 100:.1f}%"], axs[i])
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This program shows a demonstrative use of NeFSAC sampler, comparing it with a uniform sampler for the task of Fundamental matrix estimation. This is meant to be a simple pytorch-only demonstration of how to use NeFSAC, without many other fancy RANSAC tricks. For state-of-the-art accuracy and speed, we suggest to write a (simple) NeFSAC sampler in C++ and integrate it with your framework. Feel free to use the pretrained models from this repository!')
    parser.add_argument("--image1", type=str, default='media/02928139_3448003521.jpg', help='Path of the first image to use for matching')
    parser.add_argument("--image2", type=str, default='media/02085496_6952371977.jpg', help='Path of the second image to use for matching')
    parser.add_argument("--device", type=str, default='cuda', help='Which device to run RANSAC on. Specify cpu for running without GPU.')
    ops = parser.parse_args()
    with torch.no_grad():
        main(ops.image1, ops.image2, ops.device)
