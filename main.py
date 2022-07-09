from model import NeFSAC
import torch


if __name__ == '__main__':
    nefsac = NeFSAC('pretrained_models/nefsac_E_kitti.pt', sample_size=5)
    correspondences = torch.rand((100, 4))
    minimal_samples = nefsac.produce_minimal_samples(correspondences)
    print(minimal_samples.shape)
