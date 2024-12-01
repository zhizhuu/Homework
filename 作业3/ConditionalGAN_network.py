import torch
import torch.nn as nn


class Condition_GAN_Generate(nn.Module):
    def __init__(self, img_channels, condition_channels):
        super(Condition_GAN_Generate, self).__init__()
        # Encoder (Convolutional Layers)
        # 输入：img_channels + condition_channels x 256 x 256
        # 下采样
        self.down0 = nn.Sequential(
            nn.Conv2d(img_channels + condition_channels, 32, kernel_size=3, padding=1),
            # 输出：32 x 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride = 2, padding=1),
            # 输出：64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding=1),
            # 输出：128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # 输出：256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # 输出：512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            # 输出：1024 x 8 x 8
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # 上采样
        # 输入：1024 x 8 x 8
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            # 输出：1024 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # 输出：256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # 输出：128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # 输出：64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # 输出：32 x 256 x 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            # 输出：3 x 256 x 256
            nn.Tanh()
        )

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        # down sample
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        # x = self.down5(x)

        # up sample
        # x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.out(x)


# 定义判别器网络
class Condition_GAN_Discriminitor(nn.Module):
    def __init__(self, img_channels, condition_channels):
        super(Condition_GAN_Discriminitor, self).__init__()
        # Encoder (Convolutional Layers)
        # 输入：6 x 256 x 256
        # self.down0 = nn.Sequential(
        #     nn.Conv2d(img_channels + condition_channels,
        #               img_channels + condition_channels,
        #               kernel_size=3, padding=1),
        #     nn.BatchNorm2d(img_channels + condition_channels),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(img_channels + condition_channels,
        #               img_channels + condition_channels,
        #               kernel_size=3, padding=1),
        #     nn.BatchNorm2d(img_channels + condition_channels),
        #     nn.LeakyReLU(0.2, inplace=True)
        #     # 输出：6 x 256 x 256
        # )
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels + condition_channels, 16, kernel_size=3, padding=1),
            # 输出：16 x 256 x 256
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            # 输出：32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # 输出：64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # 输出：64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        # 输入图片维度是[batch size * dim * h * w ] = [batchsize * 3 * 256 * 256]
        # print(x.shape)
        x = torch.cat((x, condition), dim=1)
        # x:1 * 6 * 256 * 256
        # down sample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)


        return self.out(x)