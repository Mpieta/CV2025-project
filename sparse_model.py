import torch.nn as nn
import math


class Conv3dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(Conv3dBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual3d(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual3d, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        is_stride_1 = (stride == 1) if isinstance(stride, int) else (all(s == 1 for s in stride))
        self.use_res_connect = is_stride_1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw (pointwise)
            layers.append(Conv3dBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw (depthwise)
            Conv3dBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear (pointwise)
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SparseModel(nn.Module):
    def __init__(self, num_classes=16, sample_size=128, width_mult=1.0, mode='standard'):
        super(SparseModel, self).__init__()
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        if mode == "high_temporal":
            interverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, (1, 2, 2)],
                [6, 32, 3, (1, 2, 2)],
                [6, 64, 4, (1, 2, 2)],
                [6, 96, 3, 1],
                [6, 160, 3, (1, 2, 2)],
                [6, 320, 1, 1],
            ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [Conv3dBNReLU(3, input_channel, stride=(1, 2, 2))]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual3d(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(Conv3dBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        # x: [Batch, 3, Frames, Height, Width]
        x = self.features(x)
        # Global Average Pooling
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()