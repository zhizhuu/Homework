import torch.nn as nn

class GAN_Generate(nn.Module):
    def __init__(self):
        super(GAN_Generate, self).__init__()
        # 输入：3 x 256 x 256
        # 下采样
        self.down0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
        # 输出：64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        # 输出：128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        # 输出：256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        # 输出：512 x 64 x 64
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 上采样
        # 输入：512 x 64 x 64
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # 输出：256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # 输出：128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # 输出：64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4,  padding=1),
            # 输出：3 x 256 x 256
            nn.Tanh(inplace=True)
        )


    def forward(self, x):
        # down sample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # up sample
        x = self.up1(x)
        x = self.up2(x)

        return self.out(x)

# 定义判别器网络
class GAN_Discriminitor(nn.Module):
    def __init__(self):
        super(GAN_Discriminitor, self).__init__()
        # Encoder (Convolutional Layers)
        # 输入：3 x 256 x 256
        self.down0 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride = 2, padding=1),
            # 输出：8 x 128 x 128
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride = 2, padding=1),
            # 输出：16 x 64 x 64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # 输出：32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # 输出：64 x 16 x 16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,1),
            nn.LeakyReLU(0.2),
            # nn.Linear(64 * 16 * 16, 16 * 16),
            # nn.LeakyReLU(0.2),
            # nn.Linear(16 * 16, 64),
            # nn.LeakyReLU(0.2),
            # nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x, label):
        # down sample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        return self.out(x)
    