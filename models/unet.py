import torch
import torch.nn as nn

__all__ = ['UNet']

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

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]

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

        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], 4, stride=2, padding=1)
        self.decode_conv1 = DoubleConv2d(filters[4], filters[3])
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 4, stride=2, padding=1)
        self.decode_conv2 = DoubleConv2d(filters[3], filters[2])
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 4, stride=2, padding=1)
        self.decode_conv3 = DoubleConv2d(filters[2], filters[1])
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], 4, stride=2, padding=1)
        self.decode_conv4 = DoubleConv2d(filters[1], filters[0])

        self.output = nn.Conv2d(filters[0], n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.size()[2] % 16 == 0, "The 2nd dimension must be a multiple of 16! but is.".format(x.size()[2])
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

        up1 = self.up1(u)
        decode_conv1 = self.decode_conv1(torch.cat([encode_conv4, up1], 1))
        up2 = self.up2(decode_conv1)
        decode_conv2 = self.decode_conv2(torch.cat([encode_conv3, up2], 1))
        up3 = self.up3(decode_conv2)
        decode_conv3 = self.decode_conv3(torch.cat([encode_conv2, up3], 1))
        up4 = self.up4(decode_conv3)
        decode_conv4 = self.decode_conv4(torch.cat([encode_conv1, up4], 1))

        return self.output(decode_conv4)

if __name__ == "__main__":
    unet = UNet(3,1)

    print(unet)

    x = torch.rand([1, 3, 512, 512])

    y = unet(x)
    print(y.shape)
