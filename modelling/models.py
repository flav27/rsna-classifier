import torch
import torch.nn as nn
import kornia
import timm

import torch.nn.functional as F


class Gray(nn.Module):
    """
    Apply Image Standardization on single channel images
    """
    IMAGE_GRAY_MEAN = 0.5
    IMAGE_GRAY_STD = 0.5

    def __init__(self, ):
        super(Gray, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 1, 1, 1))
        self.register_buffer('std', torch.ones(1, 1, 1, 1))
        self.mean.data = torch.FloatTensor([self.IMAGE_GRAY_MEAN]).view(self.mean.shape)
        self.std.data = torch.FloatTensor([self.IMAGE_GRAY_STD]).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class ChannelWiseAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.drop_rate = 0.

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
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)

        return out


class PatchProcessor(torch.nn.Module):
    def __init__(self, model, device):
        super(PatchProcessor, self).__init__()
        self.device = device
        self.model = model
        self.flatten = torch.nn.Flatten()
        self.channel_wise_attention = ChannelWiseAttention(in_channels=128)
        self.fc = torch.nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(p=0.)

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
        self.extractor_resolution = (512, 256)
        self.patch_nr = 9
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
        only for 3x3
        """
        row1 = torch.cat([features_list[0], features_list[1], features_list[2]], dim=2)
        row2 = torch.cat([features_list[3], features_list[4], features_list[5]], dim=2)
        row3 = torch.cat([features_list[6], features_list[7], features_list[8]], dim=2)
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
        self.relu = nn.ReLU()
        self.cwa = ChannelWiseAttention(in_channels=24 * 12)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.cwa(x)
        x = self.flatten(x)
        return x


class MilAttention(nn.Module):
    def __init__(self, input_size):
        super(MilAttention, self).__init__()
        self.L = input_size
        self.D = input_size // 4
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, y_patches):
        batch_size, patches_nr, h_dim = y_patches.size()
        H = y_patches.view(batch_size * patches_nr, h_dim)
        A = self.attention(H)
        H = H.view(batch_size, patches_nr, h_dim)
        A = A.view(batch_size, patches_nr, self.K)
        A = F.softmax(A, dim=1)
        M = torch.sum(A * H, 1)
        return M


class RSNANet(torch.nn.Module):

    def __init__(self, device):
        super(RSNANet, self).__init__()
        self.device = device
        self.global_net = GlobalNet(device)
        self.patch_container = PatchProcessorContainer(device)
        self.attention_module = MilAttention(input_size=512)
        self.fc_patches = nn.Linear(512, 1)
        self.fc_global = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.)
        self.gray = Gray()

    def forward(self, x):
        x = self.gray(x)
        y_patch_preds, y_patch_features, feature_map = self.patch_container(x)
        y_patch_features = self.attention_module.forward(y_patch_features)
        y_patches = self.fc_patches(y_patch_features)
        global_features = self.global_net(feature_map)
        global_features = self.dropout(global_features)
        y_global = self.fc_global(torch.cat((global_features, y_patch_features), dim=1))
        features_out = torch.cat((global_features, y_patch_features), dim=1)
        return y_global, y_patches, features_out



