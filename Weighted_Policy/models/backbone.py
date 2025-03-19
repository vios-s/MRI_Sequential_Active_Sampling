import sys
import torch
import torch.nn as nn
from torchvision import models
from .fft_conv import FFTConv2d
from torch import view_as_complex
import torch.nn.functional as F
sys.path.append('../')
from data import center_crop



class MultiHeadResNet50(nn.Module):
    def __init__(self, config):
        super(MultiHeadResNet50, self).__init__()
        self.feature_map_input_layer = config.feature_map_input_layer
        # Load pre-trained ResNet50 and modify first conv layer
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.conv1 = nn.Conv2d(config.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original FC layer
        in_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Create multiple output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_prob),
                nn.Linear(in_features // 2, config.num_classes)
            ) for _ in range(config.num_label)
        ])

    def get_feature_map(self, x):
        feature_map_list = []
        for layer_num in range(self.feature_map_input_layer):
            layers = nn.Sequential(*list(self.backbone.children())[:(-1-layer_num)])
            layer_features = layers(x)
            global_avg_pooled = F.adaptive_avg_pool3d(layer_features, (256, 1, 80))
            feature_map_list.append(global_avg_pooled.permute(1, 0, 2, 3))

        concatenated_feature_maps = torch.cat(feature_map_list)
        final_feature_maps = concatenated_feature_maps.permute(1, 2, 3, 0)

        return final_feature_maps


    def forward(self, x):
        feature_maps = self.get_feature_map(x)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        output = [head(x) for head in self.heads]
        return feature_maps, F.softmax(output[0], dim=-1)


class MultiHeadSqueezeNet(nn.Module):
    def __init__(self, config):
        super(MultiHeadSqueezeNet, self).__init__()
        # Load pre-trained SqueezeNet and modify first conv layer
        self.backbone = models.squeezenet1_1(weights='DEFAULT').features
        # Modify first convolution layer
        self.backbone[0] = nn.Conv2d(config.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Get the number of output features from the last layer before classifier
        in_features = 512  # SqueezeNet typically ends with 512 channels before classification

        # Create multiple output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_prob),
                nn.Linear(in_features // 2, config.num_classes)
            ) for _ in range(config.num_label)
        ])

    def get_feature_map(self, x):
        feature_map_list = []
        for layer_num in range(self.feature_map_input_layer):
            layers = nn.Sequential(*list(self.backbone.children())[:(-1-layer_num)])
            layer_features = layers(x)
            global_avg_pooled = F.adaptive_avg_pool3d(layer_features, (256, 1, 80))
            feature_map_list.append(global_avg_pooled.permute(1, 0, 2, 3))

        concatenated_feature_maps = torch.cat(feature_map_list)
        final_feature_maps = concatenated_feature_maps.permute(1, 2, 3, 0)

        return final_feature_maps


    def forward(self, x):
        x = self.backbone(x)
        # Apply global average pooling to reduce spatial dimensions to 1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten the output feature map
        x = x.view(x.size(0), -1)  # This will now be (batch_size, 512)
        return [head(x) for head in self.heads]

class MultiHeadResNet18(nn.Module):
    def __init__(self, config):
        super(MultiHeadResNet18, self).__init__()

        # Load pre-trained ResNet50 and modify first conv layer
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.conv1 = nn.Conv2d(config.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original FC layer
        in_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Create multiple output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_prob),
                nn.Linear(in_features // 2, config.num_classes)
            ) for _ in range(config.num_label)
        ])

    def get_feature_map(self, x):
        feature_map_list = []
        for layer_num in range(self.feature_map_input_layer):
            layers = nn.Sequential(*list(self.backbone.children())[:(-1-self.feature_map_input)])
            layer_features = layers(x)
            global_avg_pooled = F.adaptive_avg_pool3d(layer_features, (256, 1, 80))
            feature_map_list.append(global_avg_pooled.permute(1, 0, 2, 3))

        concatenated_feature_maps = torch.cat(feature_map_list)
        final_feature_maps = concatenated_feature_maps.permute(1, 2, 3, 0)

        return final_feature_maps

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return [head(x) for head in self.heads]



class KspaceNetRes50(nn.Module):
    def __init__(self, config):
        super(KspaceNetRes50, self).__init__()

        self.recon_size = config.recon_size

        # FFT-based convolution layer
        self.conv_kspace = FFTConv2d(
            config.in_channel, config.in_channel, kernel_size=5, stride=1, bias=False
        )

        # Layer normalization
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=config.recon_size
        )

        # Load pre-trained ResNet50 and modify first conv layer
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone.conv1 = nn.Conv2d(config.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.maxpool = nn.Identity()

        # Remove the original fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Create multiple output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(in_features, in_features // 2),
                # nn.ReLU(),
                nn.Dropout(p=config.dropout_prob),
                nn.Linear(in_features, config.num_classes)
            ) for _ in range(config.num_label)
        ])

    def get_feature_map(self, x):

        x = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        x = torch.fft.fftn(x, dim=(-2, -1))
        # Apply FFT-based convolution
        x = self.conv_kspace(x)

        # Take absolute value and center crop
        x = x.abs()
        x = center_crop(x, self.recon_size)

        # Layer normalization
        x = self.layernorm(x)

        # Pass through ResNet50 backbone
        feature_map = self.backbone(x)

        return feature_map

    def forward(self, x):
        x = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        x = torch.fft.fftn(x, dim=(-2, -1))
        # Apply FFT-based convolution
        x = self.conv_kspace(x)

        # Take absolute value and center crop
        x = x.abs()
        x = center_crop(x, self.recon_size)

        # # Layer normalization
        # x = self.layernorm(x)

        # Pass through ResNet50 backbone
        x = self.backbone(x)
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Apply each head to the flattened output
        return [head(x) for head in self.heads]
