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
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return [head(x) for head in self.heads]

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


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),  # Does not use batch statistics: unaffected by model.eval() or model.train()
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):

    def __init__(self, config):
        """
        config:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = config.in_channel
        self.out_chans = config.out_channel
        self.chans = config.channel
        self.num_pool_layers = config.num_pool_layers
        self.drop_prob = config.dropout_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(config.in_channel, config.channel, config.dropout_prob)])
        ch = config.channel
        for _ in range(config.num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, config.dropout_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, config.dropout_prob)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(config.num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, config.dropout_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, config.dropout_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, config.out_channel, kernel_size=1),
            nn.Conv2d(config.out_channel, config.out_channel, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)