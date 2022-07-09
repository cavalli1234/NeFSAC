import torch
from torch import nn
from torch.nn.functional import relu, leaky_relu


class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 dims,
                 hidden_act=relu,
                 final_act=torch.sigmoid,
                 base_shape=(-1, )):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.hidden_act = hidden_act
        self.final_act = final_act
        self.out_shape = base_shape + (dims[-1], )

    def forward(self, x):
        original_shape = x.shape
        channels = original_shape[-1]
        x = x.view(-1, channels)
        it = 0
        for lin in self.layers:
            it += 1
            x = lin(x)
            if it < len(self.layers):
                x = self.hidden_act(x)
        return self.final_act(x).view(self.out_shape)


class NeFSAC_score_model(nn.Module):
    def __init__(self, sample_size=5, branches_out=2):
        super().__init__()
        self.mlp1 = MultiLayerPerceptron([4, 32, 32],
                                         hidden_act=leaky_relu,
                                         final_act=leaky_relu,
                                         base_shape=(-1, sample_size))
        self.mlp2 = MultiLayerPerceptron([2 * 32, 64, 64],
                                         hidden_act=leaky_relu,
                                         final_act=leaky_relu,
                                         base_shape=(-1, sample_size))
        self.mlp_fin = MultiLayerPerceptron([64, 32, branches_out],
                                            hidden_act=leaky_relu,
                                            base_shape=(-1, ))
        self.register_parameter(
            "aggr_weights",
            torch.nn.Parameter(torch.tensor([1. / branches_out] * branches_out,
                                            dtype=torch.float32),
                               requires_grad=True))

    def aggregate(self, x):
        glob = x.max(dim=-2, keepdim=True).values
        return torch.cat([x, glob.expand(x.shape)], dim=-1)

    def extract_features(self, x):
        x = self.mlp1(x)
        x = self.aggregate(x)
        x = self.mlp2(x)
        glob_x = x.max(dim=-2).values
        return glob_x

    def forward(self, x):
        xprime = torch.cat([x[..., 2:], x[..., :2]], dim=-1)
        f1 = self.extract_features(x)
        f2 = self.extract_features(xprime)
        scores = self.mlp_fin(torch.maximum(f1, f2))
        final_score = scores.detach().log().mul(
            self.aggr_weights).sum(-1).exp()
        return final_score


class NeFSAC:
    def __init__(self,
                 model_path,
                 sample_size=5,
                 branches_out=2,
                 sample_pool_size=10000,
                 output_size=500):
        self.model = torch.load(model_path)
        # self.model = NeFSAC_score_model(
        #     sample_size, branches_out).load_state_dict(state_dict)
        self.sample_pool_size = sample_pool_size
        self.output_size = output_size
        self.sample_size = sample_size

    def produce_minimal_samples(self, correspondences: torch.Tensor):
        num_correspondences = correspondences.shape[0]
        assert correspondences.shape[1] == 4
        minimal_samples_idx = torch.randint(low=0, high=num_correspondences, size=(self.sample_pool_size, self.sample_size)).to(correspondences.device)
        minimal_samples = correspondences[minimal_samples_idx]
        scores = self.model(minimal_samples).squeeze()
        best_idx = torch.topk(scores, self.output_size).indices
        return minimal_samples[best_idx]
