import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models  # Assuming you have a ResNet50 model implementation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return feature_maps, output
