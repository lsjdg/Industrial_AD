import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
import torch.nn.functional as F


class DomainRelated_Feature_Selection(nn.Module):
    def __init__(self, num_channels=256):
        super().__init__()
        self.num_channels = num_channels

        # theta for each level
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).to("cuda")
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1)).to("cuda")
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1)).to("cuda")

    def forward(self, xs, priors, learnable=True):
        features = []
        for idx, (x, prior) in enumerate(zip(xs, priors)):
            theta = 1
            if learnable:
                if idx < 3:
                    theta = torch.clamp(
                        torch.sigmoid(eval(f"self.theta{idx + 1}")) * 1.0 + 0.5,
                        max=1,
                    )
                else:
                    theta = torch.clamp(
                        torch.sigmoid(eval(f"self.theta{idx - 2}")) * 1.0 + 0.5,
                        max=1,
                    )

            b, c, h, w = x.shape
            prior_flat = prior.reshape(b, c, -1)
            prior_flat_max = prior_flat.max(dim=-1, keepdim=True)[
                0
            ]  # max returns a namedtuple
            prior_flat -= prior_flat_max

            weights = torch.softmax(prior_flat, dim=-1)
            weights = weights.reshape(b, c, h, w)

            global_inf = prior.mean(dim=(-1, -2), keepdim=True)

            inter_weights = weights * (theta + global_inf)
            x_ = x * inter_weights
            features.append(x_)

        return features


if __name__ == "__main__":
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(t.max(dim=-1, keepdim=True)[0])
