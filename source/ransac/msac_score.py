import torch

class MSACScore(object):

    def __init__(self, threshold):
        self.threshold = (3 / 2 * threshold)**2
        self.th = (3 / 2) * threshold
        self.provides_inliers = True

    def select_inliers(self, matches, models):
        scores, masks, squared_residuals = self.score(matches, models)
        best_score, best_iter = torch.max(scores, dim=0)
        return best_score, masks[best_iter]

    def score(self, matches, models):
        """
            rewrite from Graph-cut Ransac
            github.com/danini/graph-cut-ransac
            calculate the Sampson distance between a point correspondence and essential/ fundamental matrix.
            Sampson distance is the first order approximation of geometric distance, calculated from the closest correspondence
            who satisfy the F matrix.
            :param: x1: x, y, 1; x2: x', y', 1;
            M: F/E matrix
        """
        pts1 = matches[:, 0:2]
        pts2 = matches[:, 2:4]
        dev = matches.device
        assert dev == models.device

        num_pts = pts1.shape[0]
        #truncated_threshold = 3 / 2 * threshold  # wider threshold

        # get homogenous coordinates
        hom_pts1 = torch.cat((pts1, torch.ones((num_pts, 1), device=dev)), dim=-1)
        hom_pts2 = torch.cat((pts2, torch.ones((num_pts, 1), device=dev)), dim=-1)

        # calculate the sampson distance and msac scores
        M_x1_ = models.matmul(hom_pts1.transpose(-1, -2))
        M_x2_ = models.transpose(-1, -2).matmul(hom_pts2.transpose(-1, -2))
        JJ_T_ = M_x1_[:, 0] ** 2 + M_x1_[:, 1] ** 2 + M_x2_[:, 0] ** 2 + M_x2_[:, 1] ** 2
        x1_M_x2_ = hom_pts1.matmul(M_x2_)
        try:
            squared_distances = (torch.diagonal(x1_M_x2_, dim1=1, dim2=2)) ** 2 / JJ_T_
        except ValueError:
            print("error")
        masks = squared_distances < self.threshold
        # soft inliers, sum of the squared distance, while transforming the negative ones to zero by torch.clamp()
        msac_scores = torch.sum(torch.clamp(1 - squared_distances / self.threshold, min=0), dim=-1)

        # folowing c++
        squared_residuals = torch.sum(torch.where(squared_distances>=self.threshold, torch.zeros_like(squared_distances), squared_distances), dim=-1)
        inlier_number = torch.sum(squared_distances.squeeze(0) < self.threshold, dim=-1)
        # score = (-squared_residuals + inlier_number * self.threshold)/self.threshold

        return torch.nan_to_num(msac_scores), masks, squared_residuals
        #return torch.sum(masks, dim=-1), masks
