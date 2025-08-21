import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRelated_Feature_Selection(nn.Module):
    def __init__(self, num_channels=256):
        super(DomainRelated_Feature_Selection, self).__init__()
        # Use ParameterList for safer and clearer parameter management
        self.thetas = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, num_channels, 1, 1)),
                nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1)),
                nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1)),
            ]
        )

    def forward(self, xs, priors, learnable=True, conv=False, max=True):
        features = []
        for idx, (x, prior) in enumerate(zip(xs, priors)):
            theta = 1
            if learnable:
                if idx < 3:
                    # Access parameters safely from the list
                    current_theta = self.thetas[idx]
                    theta = torch.clamp(torch.sigmoid(current_theta) * 1.0 + 0.5, max=1)

            b, c, h, w = x.shape
            if not conv:
                prior_flat = prior.view(b, c, -1)
                if max:
                    prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
                    prior_flat = prior_flat - prior_flat_
                weights = F.softmax(prior_flat, dim=-1)
                weights = weights.view(b, c, h, w)

                global_inf = prior.mean(dim=(-2, -1), keepdim=True)

                inter_weights = weights * (theta + global_inf)

                x_ = x * inter_weights
                features.append(x_)
            else:
                #  generated in a convolutional way
                pass

        return features


def domain_related_feature_selection(xs, priors, max=True):
    features_list = []
    theta = 1
    for x, prior in zip(xs, priors):
        b, c, h, w = x.shape

        prior_flat = prior.view(b, c, -1)
        if max:
            prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
            prior_flat = prior_flat - prior_flat_
        weights = F.softmax(prior_flat, dim=-1)
        weights = weights.view(b, c, h, w)

        global_inf = prior.mean(dim=(-2, -1), keepdim=True)

        inter_weights = weights * (theta + global_inf)

        x_ = x * inter_weights
        features_list.append(x_)

    return features_list
