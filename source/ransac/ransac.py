import random
import cv2
import numpy as np
import torch
from .errors import squared_sampson_distance


def ransac_fundamental_matrix(source_points, destination_points,
                              inlier_outlier_threshold, confidence,
                              maximum_iterations,
                              filter_model=None,
                              device="cuda",
                              seed=True):

    if seed:
        random.seed(a=0, version=2)
    point_number = source_points.shape[0]
    iterations = 0
    squared_threshold = inlier_outlier_threshold**2
    sample_size = 7
    maximum_unsuccessful_sampling = 1000
    estimated_fundamental_matrix = []
    final_best_inliers = []
    used_samples = set()

    # The ratio of samples kept
    sample_keeping_ratio = 1.0
    # The index of the best sample

    # In case the filtering model is not None
    if filter_model is not None:
        correspondences = torch.cat([
            torch.tensor(source_points,
                         dtype=torch.float32),
            torch.tensor(destination_points,
                         dtype=torch.float32)
        ],
                           dim=-1).to(device)
        samples = filter_model.produce_minimal_samples(correspondences)[1].cpu().numpy()
        samples_iter = (list(s) for s in samples)


    # Main RANSAC loop
    while (iterations < maximum_iterations):
        iterations += 1

        if filter_model is None:
            # Selecting a minimal sample
            unsuccessful_sampling = 0
            while unsuccessful_sampling < maximum_unsuccessful_sampling:
                sample = random.sample(range(point_number), sample_size)
                sample.sort()
                if tuple(sample) not in used_samples:
                    break
                unsuccessful_sampling += 1

            if unsuccessful_sampling >= maximum_unsuccessful_sampling:
                break
        else:
            # If the filtering model is set, get the next sample from the list
            # generated earlier.
            try:
                sample = next(samples_iter)
            except StopIteration:
                # If there is not next sample, we can terminate.
                # Note that this is not nice since the RANSAC termination criterain might not
                # have been triggered. However, it works if the maximum iteration number
                # is reasonably large.
                break
        used_samples.add(tuple(sample))

        # Estimate fundamental matrices from the selected sample by the
        # seven-point algorithm.
        Fs, _ = cv2.findFundamentalMat(source_points[sample, :],
                                       destination_points[sample, :],
                                       cv2.FM_7POINT)

        # Check if the estimation was successful
        if Fs is None:
            continue

        # Check if at least a single model is returned
        if range(round(Fs.shape[0] / 3)) == 0:
            continue

        best_inliers = []
        best_F = []

        # Iterating through the estimated fundamental matrices and selecting
        # the one with the most inliers. This is what would happen in RANSAC
        for F_idx in range(round(Fs.shape[0] / 3)):
            F = Fs[F_idx * 3:(F_idx + 1) * 3, :]

            # Count inliers
            inliers = []
            for point_idx in range(point_number):
                squared_residual = squared_sampson_distance(
                    source_points[point_idx, :],
                    destination_points[point_idx, :], F)

                if squared_residual < squared_threshold:
                    inliers.append(point_idx)

            if len(best_inliers) < len(inliers):
                best_inliers = inliers
                best_F = F

        # Testing if the best model has more inliers than the minimal sample
        if len(best_inliers) < sample_size:
            continue

        if len(best_inliers) > len(final_best_inliers):
            estimated_fundamental_matrix = best_F
            final_best_inliers = best_inliers

    return estimated_fundamental_matrix, final_best_inliers


def ransac_essential_matrix(source_points,
                            destination_points,
                            inlier_outlier_threshold,
                            confidence,
                            maximum_iterations,
                            K1,
                            K2,
                            filter_model=None,
                            device="cuda",
                            seed=True):

    # Fixing the random seed
    if seed:
        random.seed(a=0, version=2)
    # The number of data points
    point_number = source_points.shape[0]
    # The number of iterations that will be increased
    iterations = 0
    # The sample size for estimating an essential matrix
    sample_size = 5
    # The set of samples used.
    used_samples = set()
    # The maximum number of sampling without success.
    maximum_unsuccessful_sampling = 1000
    # The final so-far-the-best essential matrix
    estimated_essential_matrix = []
    # The inliers of the so-far-the-best model
    final_best_inliers = []
    # The MSAC of the so-far-the-best model
    final_best_score = 0
    # The threshold normalizer calculated from the focal lengths
    threshold_normalizer = (max(K1[0, 0], K1[1, 1]) +
                            max(K2[0, 0], K2[1, 1])) / 2.0
    # The normalized threshold that will be used when selecting the inliers
    normalized_threshold = inlier_outlier_threshold / threshold_normalizer
    # The squared threshold just so we don't have to get the square-root of the residual
    squared_threshold = normalized_threshold**2

    # Normalize the points by the camera matrices
    normalized_source_points = cv2.undistortPoints(source_points[:, None], K1,
                                                   0)
    normalized_destination_points = cv2.undistortPoints(
        destination_points[:, None], K2, 0)

    # In case the filtering model is not None
    if filter_model is not None:
        correspondences = torch.cat([
            torch.tensor(source_points,
                         dtype=torch.float32),
            torch.tensor(destination_points,
                         dtype=torch.float32)
        ],
                           dim=-1).to(device)
        samples = filter_model.produce_minimal_samples(correspondences)[1].cpu().numpy()
        samples_iter = (list(s) for s in samples)

    # Main RANSAC loop
    while (iterations < maximum_iterations):
        # Increase the iteration number
        iterations += 1

        # Do this if the filtering model is not set and, thus,
        # the samples have not been created earlier.
        if filter_model is None:
            unsuccessful_sampling = 0
            # Iterate until we have enough samples or there is nothing promising
            while unsuccessful_sampling < maximum_unsuccessful_sampling:
                # Do sampling uniformly randomly
                sample = random.sample(range(point_number), sample_size)
                # Sorting the indices so it can be stored.
                # This is not too practical for RANSAC since it takes time, but
                # it is important for the training to generate a sample only once.
                sample.sort()
                # Checking if we have already used this sample
                # If it is a new one, let's try it out.
                if tuple(sample) not in used_samples:
                    break
                # Otherwise, increase the number of unsuccessful samples
                unsuccessful_sampling += 1
                # Break out from the loop if we have not seen a new sample for a while
                if unsuccessful_sampling >= maximum_unsuccessful_sampling:
                    break
        else:
            # If the filtering model is set, get the next sample from the list
            # generated earlier.
            try:
                sample = next(samples_iter)
            except StopIteration:
                # If there is not next sample, we can terminate.
                # Note that this is not nice since the RANSAC termination criterain might not
                # have been triggered. However, it works if the maximum iteration number
                # is reasonably large.
                break
        used_samples.add(tuple(sample))

        # Estimate fundamental matrices from the selected sample by the
        # seven-point algorithm.
        Es, _ = cv2.findEssentialMat(normalized_source_points[sample, :],
                                     normalized_destination_points[sample, :],
                                     np.identity(3),
                                     cv2.RANSAC,
                                     threshold=1e10)

        # Check if the estimation was successful
        if Es is None:
            continue

        # Check if at least a single model is returned
        if range(round(Es.shape[0] / 3)) == 0:
            continue

        # MSAC score of the best model from the
        best_score = 0.0
        best_inliers = []
        best_E = []

        # Iterating through the estimated fundamental matrices and selecting
        # the one with the most inliers. This is what would happen in RANSAC
        for E_idx in range(round(Es.shape[0] / 3)):
            E = Es[E_idx * 3:(E_idx + 1) * 3, :]

            # Count inliers
            inliers = []
            score = 0.0
            for point_idx in range(point_number):
                squared_residual = squared_sampson_distance(
                    normalized_source_points[point_idx, :][0],
                    normalized_destination_points[point_idx, :][0], E)

                if squared_residual < squared_threshold:
                    score += 1 - squared_residual / squared_threshold
                    inliers.append(point_idx)

            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_E = E

        # Testing if the best model has more inliers than the minimal sample
        if len(best_inliers) < sample_size:
            continue

        if best_score > final_best_score:
            estimated_essential_matrix = best_E
            final_best_inliers = best_inliers
            final_best_score = best_score

    try:
        # Do a final model refitting on all inliers.
        # OpenCV RANSAC acts weirdly and it unfortunately does not have
        # simply LSQ.
        Es, _ = cv2.findEssentialMat(
            normalized_source_points[final_best_inliers, :],
            normalized_destination_points[final_best_inliers, :],
            np.identity(3),
            cv2.LMEDS,
            threshold=normalized_threshold)

        # Replace the model parameters if the estimation was successfull
        if Es.shape[0] == 3 and Es.shape[1] == 3:
            estimated_essential_matrix = Es
    except:
        pass

    return estimated_essential_matrix, final_best_inliers

