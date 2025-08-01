import torch
import torch.nn as nn
from models.backbones import *
import math
import os
from utils.sampling import coreset_sampling


# N: batch size
# N': memory bank size
# query: query size
# D: target_dim
# |A|: patch collection size
# |P|: patch_size


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone=resnet18,
        memory_bank_path="./memory_bank/resnet18",
        device="cuda",
        patch_size=3,
        stride=2,
        coverage=0.01,
    ):
        super().__init__()
        self.backbone = backbone
        self.memory_bank = None
        self.memory_bank_path = memory_bank_path
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.coverage = coverage

        self.backbone = backbone.to(self.device)

        self.layer2_output, self.layer3_output = None, None

        self.backbone.layer2.register_forward_hook(self._hook_layer2)
        self.backbone.layer3.register_forward_hook(self._hook_layer3)

    def _hook_layer2(self, module, input, output):
        self.layer2_output = output  # (B, 128, 28, 28)

    def _hook_layer3(self, module, input, output):
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bicubic", align_corners=False
        )  # scale_factor: Upsample to match layer2
        self.layer3_output = output  # (B, 256, 28, 28)

    def _embed(self, x):
        """
        Extracts and aggregates patch features from a batch of images using average pooling.
        This is a vectorized and more efficient implementation.
        """
        # 1. Run backbone to get intermediate features via hooks
        self.backbone(x)

        # 2. Concatenate features from different layers
        combined_features = torch.cat([self.layer2_output, self.layer3_output], dim=1)

        # 3. Apply average pooling to aggregate local features
        # This replaces the manual iteration in the original `get_local_feature`.
        padding = (self.patch_size - 1) // 2
        patch_features = nn.functional.avg_pool2d(
            combined_features,
            kernel_size=self.patch_size,
            stride=1,
            padding=padding,
        )

        # 4. Subsample the feature map according to the specified stride
        sampled_features = patch_features[:, :, :: self.stride, :: self.stride]

        # 5. Reshape for coreset sampling and distance calculation
        # [B, D, H', W'] -> [B, D, A] -> [B, A, D]
        # where A = H' * W' is the number of patches
        patches = sampled_features.flatten(2).permute(0, 2, 1)

        return patches

    def extract_features(self, train_batch):
        """
        Extracts patch features from a batch of images.
        """
        patches = self._embed(train_batch)  # [B, A, D]
        B, A, D = patches.shape
        return patches.reshape(B * A, D)

    def build_memory_bank(self, all_features, path):
        """
        Builds the memory bank using coreset sampling on all training features
        and saves it to a single file.
        """
        coreset = coreset_sampling(
            all_features, int(all_features.size(0) * self.coverage)
        )
        self.memory_bank = coreset.to(self.device)

        os.makedirs(self.memory_bank_path, exist_ok=True)
        torch.save(coreset.cpu(), path)

    def load_memory(self, path):
        """
        Loads the memory bank from a single file.
        """
        self.memory_bank = torch.load(path, map_location=self.device)

    def forward(self, x):
        # extract query patches
        q = self._embed(x)  # [B, A, D]
        B, A, D = q.shape
        # flatten patches: [B*A, D]
        q_flat = q.reshape(B * A, D)
        # compute distances to memory: [B*A, N']
        dists = torch.cdist(q_flat, self.memory_bank)
        # min distance per patch: [B*A]
        min_patch = dists.min(dim=1).values
        # reshape to [B, A]
        min_patch = min_patch.view(B, A)
        # anomaly score: max over patches
        scores = min_patch.max(dim=1).values

        return scores
