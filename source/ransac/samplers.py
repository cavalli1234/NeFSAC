import torch


class UniformSampler:
    """
    A simple, uniform sampler.

    It produces uniformly random samples without repetitions.
    """
    def __init__(self, n_iters, n_samples_per_iter):
        self.n_iters = n_iters
        self.n_samples_per_iter = n_samples_per_iter
        self.n_iters_per_batch = n_iters

    def sample(self, data, *args, device, **kwargs):
        k = data.shape[0]
        # The first sample is easy to take.
        samples = torch.randint(low=0,
                                high=k - 1,
                                size=(self.n_iters, 1),
                                device=device)
        # Now extract all the rest without repetitions.
        while samples.shape[1] < self.n_samples_per_iter:
            k = k - 1
            new_s = torch.randint(low=0,
                                  high=k - 1,
                                  size=(self.n_iters, 1),
                                  device=device)
            new_s = new_s + (new_s >= samples - torch.arange(
                samples.shape[1], device=device).unsqueeze(0)).long().sum(
                    dim=1, keepdim=True)
            samples = torch.sort(torch.cat([samples, new_s], dim=1), dim=1)[0]

        return samples


class NeFSACSampler:
    """
    A simple implementation of the NeFSAC sampler. This shows how to use a pretrained NeFSAC model for usage in RANSAC.
    """
    def __init__(self, n_iters, n_samples_per_iter, model_path, dev='cuda', keep_rate=0.05, effective_iters_multiplier=2.):
        """
        Args
            n_iters: number of minimal samples to output at every call of self.sample
            n_samples_per_iter: size of the minimal sample (set 5 for E estimation, 7 for F estimation, or according to the task)
            model_path: path to a torchscript pretrained model
            dev: device to load NeFSAC to
            keep_rate: NeFSAC will output the best keep_rate minimal samples from the base sampler.
                       This means that int(n_iters/keep_rate) minimal samples will be processed to
                       output exactly n_iters in the end.
            effective_iters_multiplier: How many iterations RANSAC should count for each iteration from NeFSAC.
                                        Since minimal samples from NeFSAC are not uniformly at random,
                                        this parameter can be used to influence the termination criterion in RANSAC.
        """
        self.n_iters = n_iters
        self.n_samples_per_iter = n_samples_per_iter
        self.n_iters_per_batch = int(n_iters * effective_iters_multiplier)
        # Note that the base sampler could also be different than Uniform (e.g. PROSAC)
        self.base_sampler = UniformSampler(n_iters=int(self.n_iters / keep_rate),
                                           n_samples_per_iter=self.n_samples_per_iter)
        self.model = torch.jit.load(model_path, map_location=dev)
        self.norm_array = None
        self.device = dev

    def set_F_normalization(self, im1shape, im2shape):
        """
        Since pixel coordinates can range very wildly in F estimation, they are normalized in NeFSAC.
            Note that in E estimation we expect the correspondences in normalized camera coordinates,
            so this step is only needed for F estimation.
        """
        self.norm_array = torch.tensor([im1shape[1], im1shape[0], im2shape[1], im2shape[0]], device=self.device)
        assert self.n_samples_per_iter == 7, "F normalization is required only for fundamental matrix estimation"

    def sample(self, data, *args, device, **kwargs):
        # First take **many** samples according to the base sampler
        samples = self.base_sampler.sample(data, *args, device=device, **kwargs)
        # Take the actual data points to produce the batch of minimal samples with shape (B, M, C)
        minimal_samples = data[samples]
        # Normalize if needed to bring their range to [-1, 1]
        if self.norm_array is not None:
            minimal_samples = minimal_samples / self.norm_array * 2. - 1.
        # Run the model
        scores = self.model(minimal_samples).squeeze()
        # Find the best scores, and only take those minimal samples
        best_samples_idx = torch.topk(scores, self.n_iters).indices
        # Return the indices of the most promising minimal samples
        return samples[best_samples_idx]


