from model import NeFSAC
from ransac import ransac_fundamental_matrix
import numpy as np

if __name__ == '__main__':
    # Example usage
    nefsac = NeFSAC('pretrained_models/nefsac_F_kitti.pt', sample_size=7)
    source_pts = np.random.rand(100, 2) * 256
    dest_pts = np.random.rand(100, 2) * 256

    F, inliers = ransac_fundamental_matrix(source_pts, dest_pts, 5., 0.999,
                                           1000, nefsac, "cpu")
    print(F)
    print(inliers)
