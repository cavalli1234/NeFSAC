import torch


def ransac(data, sampler, solver_func, scoring_func, eps=1e-3, max_iters=10000):
    """
    Perform torch-compatible configurable RANSAC in batches

    Args:
        data: a Tensor of data points of shape (N, C) where N is the number of data points, and C their dimensionality
        sampler: an object implementing:
            a method sample: given the data Tensor, it returns indices of shape (B, M) where B is a batch
                             of iterations to perform in parallel, and M is the minimal sample size
            an integer property n_iters_per_batch: it specifies how many iterations have been performed after
                                                   processing the full sample. Can be different from B.
        solver_func: a callable that takes as input a batch of B samples of M data points (B, M, C) and
                     returns a tensor of models. It can return any number of models, not necessarily B.
                     It should also accept values of M which are not minimal, but it does not need to be robust.
        scoring_func: a callable that takes as input the data Tensor and a batch of models, and returns
                      a single score (float) and a boolean mask of the inliers according to the best model
        eps: the accepted probability of termination without finding a good model
        max_iters: maximum number of iterations before forcefully stopping
    Returns:
        a boolean tensor of size (B,) to be used as an inlier mask to the data input Tensor
        note: for the final fit model, the solver_func can be used on the inliers.
    """
    iters_done = 0
    best_inliers = None
    best_inlier_rate = 0.
    best_score = 0.
    while iters_done < max_iters:
        n_pts, n_chs = data.shape
        samples = sampler.sample(data, device=data.device)  # (n_iters, n_samples)
        n_iters, n_samples = samples.shape
        sampled_data = data[samples]  # (n_iters, n_samples, n_chs)
        models = solver_func(sampled_data)  # (n_iters, n_mod_chs)
        score, inliers = scoring_func(data, models)
        # Check that the scoring_func is compliant
        assert inliers.shape == (n_pts,), f"Expected the inliers mask to have shape {(n_pts,)} but found shape {inliers.shape}."

        if score > best_score:
            # Perform Local Optimization
            LO_model = solver_func(data[inliers].unsqueeze(0))
            LO_score, LO_inliers = scoring_func(data, LO_model)
            if LO_score > score:
                score = LO_score
                inliers = LO_inliers

            # Register best model
            best_score = score
            best_inliers = inliers
            best_inlier_rate = best_inliers.float().mean()

        iters_done += sampler.n_iters_per_batch
        p = (1.0 - best_inlier_rate ** n_samples) ** iters_done
        #print(p, best_inlier_rate, score, LO_score, best_score, n_samples, iters_done)
        if p < eps:
            return best_inliers
    return best_inliers
