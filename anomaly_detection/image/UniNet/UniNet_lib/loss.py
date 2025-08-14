import torch
import torch.nn.functional as F


def loss(b, a, T, margin, lmbda, mask=None, stop_grad=False):
    """
    b: List of teacher features
    a: List of student features
    T: Temperature coefficient
    margin: Hyperparameter for controlling the boundary
    lmbda: Hyperparameter for balancing loss
    mask: Binary mask, where 0 for normal and 1 for abnormal
    """
    margin_loss_n = 0.0
    margin_loss_an = 0.0
    contra_loss = 0.0

    for i in range(len(a)):
        s_ = a[i]
        t_ = b[i].detach() if stop_grad else b[i]

        n, c, h, w = s_.shape

        # view() : shape 변경
        s = s_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)
        t = t_.view(n, c, -1).transpose(1, 2)

        s_norm = F.normalize(s, p=2, d=2)
        t_norm = F.normalize(t, p=2, d=2)

        cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=2)
        cos_loss = cos_loss.mean()

        simi = torch.matmul(
            s_norm,
            t_norm.transpose(1, 2),
        )
        simi /= T
        simi = torch.exp(simi)
        simi_sum = simi.sum(dim=2, keepdim=True)
        simi = simi / (simi_sum + 1e-8)
        diag_simi = torch.diagonal(simi, dim1=1, dim2=2)

        # for unsupervised, one class (only normal)
        if mask is None:
            contra_loss = -torch.log(diag_simi + 1e-8).mean()
            margin_loss_n = F.relu(margin - diag_simi).mean()

        margin_loss = margin_loss_n + margin_loss_an
        loss = lmbda * cos_loss + (1 - lmbda) * contra_loss + margin_loss

        return loss
