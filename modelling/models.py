import torch
import torch.nn as nn
import kornia
import timm


class Gray(nn.Module):

    """
    Apply Image Standardization using the values calculated for the RSNA Screening Mammography Dataset
    """
    IMAGE_GRAY_MEAN = 0.1735
    IMAGE_GRAY_STD = 0.1986

    def __init__(self, ):
        super(Gray, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 1, 1, 1))
        self.register_buffer('std', torch.ones(1, 1, 1, 1))
        self.mean.data = torch.FloatTensor([self.IMAGE_GRAY_MEAN]).view(self.mean.shape)
        self.std.data = torch.FloatTensor([self.IMAGE_GRAY_STD]).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_size, attention_size, dropout_rate=0.5):
        super(AttentionLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, attention_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(attention_size, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        attention = torch.nn.functional.softmax(self.fc2(x), dim=1)
        return attention


class ChannelWiseAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.drop_rate = 0.5

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        weights = torch.softmax(self.fc(x), dim=2)
        x = (x * weights).sum(dim=2)
        x = x.view(b, c, 1, 1)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=1, padding=2):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)

        return out


class PatchProcessor(torch.nn.Module):
    def __init__(self, model, device):
        super(PatchProcessor, self).__init__()
        self.device = device
        self.model = model
        self.flatten = torch.nn.Flatten()
        self.channel_wise_attention = ChannelWiseAttention(in_channels=64)
        self.fc = torch.nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        feature_map = self.model(x)[-1]
        y_patch_features = self.channel_wise_attention(feature_map)
        y_patch_features = self.flatten(y_patch_features)
        y_patch_features = self.dropout(y_patch_features)
        y_pred = self.fc(y_patch_features)
        return y_pred, y_patch_features, feature_map


class PatchProcessorContainer(torch.nn.Module):
    def __init__(self, device):
        super(PatchProcessorContainer, self).__init__()
        """
        configured for 6*256x3*256
        """
        self.extractor_resolution = (256, 256)
        self.patch_nr = 18
        self.model = timm.create_model("resnet18d", pretrained=True, in_chans=1,
                                       features_only=True, out_indices=[4])
        self.device = device
        self.patch_extractor = kornia.contrib.ExtractTensorPatches(window_size=self.extractor_resolution,
                                                                   stride=self.extractor_resolution)
        self.patch_processors = self.create_patch_processors()

    def create_patch_processors(self):
        net_list = torch.nn.ModuleList()
        for i in range(0, self.patch_nr):
            net = PatchProcessor(self.model, device=self.device)
            net_list.append(net)
        return net_list

    @staticmethod
    def reassemble_feature_map(features_list):
        """
        only for 6x3
        """
        row1 = torch.cat([features_list[0], features_list[1], features_list[2],
                          features_list[3], features_list[4], features_list[5]], dim=2)
        row2 = torch.cat([features_list[6], features_list[7], features_list[8],
                          features_list[9], features_list[10], features_list[11]], dim=2)
        row3 = torch.cat([features_list[12], features_list[13], features_list[14],
                          features_list[15], features_list[16], features_list[17]], dim=2)
        feature_map = torch.cat([row1, row2, row3], dim=3)
        return feature_map

    def forward(self, x):
        patches = self.patch_extractor(x)
        features_map_list = []
        y_patch_features_list = []
        y_patch_pred_list = []
        for i in range(0, self.patch_nr):
            y_patch_pred, y_patch_features, features_map = self.patch_processors[i](patches[:, i, :, :, :])
            features_map_list.append(features_map)
            y_patch_features_list.append(y_patch_features)
            y_patch_pred_list.append(y_patch_pred)
        feature_map = self.reassemble_feature_map(features_map_list)
        y_patch_features = torch.stack(y_patch_features_list, dim=1)
        y_patch_preds = torch.stack(y_patch_pred_list, dim=1)
        return y_patch_preds, y_patch_features, feature_map


class GlobalNet(torch.nn.Module):

    def __init__(self, device):
        super(GlobalNet, self).__init__()
        self.device = device
        self.conv = ResBlock(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.ln = torch.nn.LayerNorm((512, 24, 12))
        self.relu = nn.ReLU(inplace=True)

        self.channel_wise_attention = ChannelWiseAttention(in_channels=24 * 12)
        self.flatten = torch.nn.Flatten()
        self.fc = nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.channel_wise_attention(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class RSNANet(torch.nn.Module):

    def __init__(self, device):
        super(RSNANet, self).__init__()
        self.device = device
        self.global_net = GlobalNet(device)
        self.patch_container = PatchProcessorContainer(device)
        self.attention_layer = AttentionLayer(input_size=512, attention_size=64)
        self.gray = Gray()

    def forward(self, x):
        x = self.gray(x)
        y_patch_preds, y_patch_features, feature_map = self.patch_container(x)
        attention_scores = self.attention_layer(y_patch_features)
        weighted_patches = attention_scores * y_patch_preds
        y_patches = weighted_patches.sum(dim=1)
        y_global = self.global_net(feature_map)
        return y_global, y_patches

