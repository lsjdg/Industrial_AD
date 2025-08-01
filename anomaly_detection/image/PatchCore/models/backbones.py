from torchvision import models

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
wideresnet50 = models.wide_resnet50_2(
    weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
)
