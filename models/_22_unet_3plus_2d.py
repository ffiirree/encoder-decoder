import torch
import torchsummary
import torch.nn as nn

__all__ = ['UNet3Plus2Decoders']

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class UNet3Plus2Decoders(nn.Module):
    def __init__(self, n_channels, n_classes, n_ex_channels, filters = [64, 128, 256, 512, 1024]):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encode_conv1 = DoubleConv2d(n_channels, filters[0])
        self.down1 = nn.MaxPool2d(2)
        self.encode_conv2 = DoubleConv2d(filters[0], filters[1])
        self.down2 = nn.MaxPool2d(2)
        self.encode_conv3 = DoubleConv2d(filters[1], filters[2])
        self.down3 = nn.MaxPool2d(2)
        self.encode_conv4 = DoubleConv2d(filters[2], filters[3])
        self.down4 = nn.MaxPool2d(2)

        self.u = DoubleConv2d(filters[3], filters[4])

        # 1
        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=4, stride=2, padding=1)
        self.decode_conv1 = DoubleConv2d(filters[3] * 2 + filters[0] + filters[1] + filters[2], filters[3])
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=4, stride=2, padding=1)
        self.decode_conv2 = DoubleConv2d(filters[2] * 2 + filters[0] + filters[1] + filters[4], filters[2])
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=4, stride=2, padding=1)
        self.decode_conv3 = DoubleConv2d(filters[1] * 2 + filters[0] + filters[4] + filters[3], filters[1])
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=4, stride=2, padding=1)
        self.decode_conv4 = DoubleConv2d(filters[0] * 2 + filters[4] + filters[3] + filters[2], filters[0])

        self.mask = nn.Conv2d(filters[0], 1, 3, padding=1)

        # ex decoder
        self.ex_up1 = nn.ConvTranspose2d(filters[4], filters[3], 4, stride=2, padding=1)
        self.ex_decode_conv1 = DoubleConv2d(filters[3], filters[3])
        self.ex_up2 = nn.ConvTranspose2d(filters[3], filters[2], 4, stride=2, padding=1)
        self.ex_decode_conv2 = DoubleConv2d(filters[2], filters[2])
        self.ex_up3 = nn.ConvTranspose2d(filters[2], filters[1], 4, stride=2, padding=1)
        self.ex_decode_conv3 = DoubleConv2d(filters[1], filters[1])
        self.ex_up4 = nn.ConvTranspose2d(filters[1], filters[0], 4, stride=2, padding=1)
        self.ex_decode_conv4 = DoubleConv2d(filters[0], filters[0])

        self.re = nn.Conv2d(filters[0], n_ex_channels, kernel_size=3, padding=1)

        # utils
        self.down_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.down_8 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        assert x.size()[2] % 16 == 0, "The 2nd dimension must be a multiple of 16! but is {}.".format(x.size()[2])
        assert x.size()[3] % 16 == 0, "The 3rd dimension must be a multiple of 16, but is {}.".format(x.size()[3])

        encode_conv1 = self.encode_conv1(x)
        down1 = self.down1(encode_conv1)

        encode_conv2 = self.encode_conv2(down1)
        down2 = self.down2(encode_conv2)

        encode_conv3 = self.encode_conv3(down2)
        down3 = self.down3(encode_conv3)

        encode_conv4 = self.encode_conv4(down3)
        down4 = self.down4(encode_conv4)

        u = self.u(down4)

        # ex decoder
        ex_up1 = self.ex_up1(u)
        ex_decode_conv1 = self.ex_decode_conv1(ex_up1)
        ex_up2 = self.ex_up2(ex_decode_conv1)
        ex_decode_conv2 = self.ex_decode_conv2(ex_up2)
        ex_up3 = self.ex_up3(ex_decode_conv2)
        ex_decode_conv3 = self.ex_decode_conv3(ex_up3)
        ex_up4 = self.ex_up4(ex_decode_conv3)
        ex_decode_conv4 = self.ex_decode_conv4(ex_up4)

        # 1
        up1 = self.up1(u)
        decode_conv1 = self.decode_conv1(torch.cat([
            up1,
            self.down_8(encode_conv1),
            self.down_4(encode_conv2),
            self.down_2(encode_conv3),
            encode_conv4,
            ], 1))

        up2 = self.up2(decode_conv1)
        decode_conv2 = self.decode_conv2(torch.cat([
            up2,
            self.down_4(encode_conv1),
            self.down_2(encode_conv2),
            encode_conv3,
            self.up_4(u)
            ], 1))

        up3 = self.up3(decode_conv2)
        decode_conv3 = self.decode_conv3(torch.cat([
            up3,
            self.down_2(encode_conv1),
            encode_conv2,
            self.up_8(u),
            self.up_4(decode_conv1),
            ], 1))

        up4 = self.up4(decode_conv3)
        decode_conv4 = self.decode_conv4(torch.cat([
            up4,
            encode_conv1,
            self.up_16(u),
            self.up_8(decode_conv1),
            self.up_4(decode_conv2),
            ], 1))

        return self.mask(decode_conv4), self.re(ex_decode_conv4)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3Plus2Decoders(3, 1, n_ex_channels=3, filters=[32, 64, 128, 256, 512])
    if device == torch.device('cuda'):
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)
    torchsummary.summary(model, (3, 512, 512))
