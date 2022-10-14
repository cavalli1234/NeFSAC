import torch


class ScaledRigidAlignmentSolver:
    def __init__(self, thr):
        self.thr = thr

    def fit(self, data, *args, **kwargs):
        if data.ndim == 2:
            data.unsqueeze_(0)
        bsize, npts, nchs = data.shape
        assert nchs % 2 == 0
        pc1 = data[..., :nchs//2]
        pc2 = data[...,  nchs//2:]
        ch = nchs // 2

        def normalize(pc):
            m = pc.mean(dim=1, keepdim=True)
            pc = pc - m
            scale = pc.norm(dim=-1).std(dim=1)
            pc = pc.div(scale.view(bsize, 1, 1))
            return pc, m, scale
        pc1, m1, s1 = normalize(pc1)
        pc2, m2, s2 = normalize(pc2)
        cov = pc1.transpose(1, 2) @ pc2
        u, s, vt = cov.svd()
        v = vt.transpose(2, 1)
        d = (u @ v).det().sign()
        D = torch.eye(ch, dtype=cov.dtype, device=cov.device).repeat(bsize, 1, 1)
        D[:, -1, -1] = d
        R = u @ D @ v
        # pc2 = (pc1 - m1) @ R * s + m2
        models = torch.cat([m1.view(bsize, ch), m2.view(bsize, ch),
                            (s2/s1).view(bsize, 1), R.view(bsize, ch ** 2)], dim=-1)
        return models

    def score(self, models, data_samples):
        n_samples, n_ch_double = data_samples.shape
        n_ch = n_ch_double // 2
        n_models, n_mod_ch = models.shape
        assert n_mod_ch == n_ch_double + 1 + n_ch ** 2
        m1, m2, s21, R = torch.split(models.unsqueeze(1), [n_ch, n_ch, 1, n_ch ** 2], dim=2)
        R = R.view(n_models, n_ch, n_ch)
        pc1, pc2 = torch.split(data_samples.unsqueeze(0), [n_ch, n_ch], dim=2)
        pc1_mapped = (pc1 - m1) @ R * s21 + m2
        assert pc1_mapped.shape == (n_models, n_samples, n_ch), \
                f"Expected pc1_mapped to have shape {(n_models, n_samples, n_ch)} but found shape {pc1_mapped.shape}."
        errors_matrix = (pc2 - pc1_mapped).norm(dim=-1)
        inliers_matrix = (errors_matrix < self.thr).long()
        return inliers_matrix[torch.argmax(inliers_matrix.sum(dim=1))]
