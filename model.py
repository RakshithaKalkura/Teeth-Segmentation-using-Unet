import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 1
        out_channels = 1

        # Contracting path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)  # Input: (batch_size, 1, 512, 512), Output: (batch_size, 64, 512, 512)
        self.encoder2 = self.conv_block(64, 128)          # Input: (batch_size, 64, 256, 256), Output: (batch_size, 128, 256, 256)
        self.encoder3 = self.conv_block(128, 256)         # Input: (batch_size, 128, 128, 128), Output: (batch_size, 256, 128, 128)
        self.encoder4 = self.conv_block(256, 512)         # Input: (batch_size, 256, 64, 64), Output: (batch_size, 512, 64, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)      # Input: (batch_size, 512, 32, 32), Output: (batch_size, 1024, 32, 32)

        # Expanding path (Decoder)
        self.decoder4 = self.upconv_block(1024, 512)      # Input: (batch_size, 1024, 32, 32), Output: (batch_size, 512, 64, 64)
        self.decoder3 = self.upconv_block(1024, 256)      # Input: (batch_size, 1024, 64, 64), Output: (batch_size, 256, 128, 128)
        self.decoder2 = self.upconv_block(512, 128)       # Input: (batch_size, 512, 128, 128), Output: (batch_size, 128, 256, 256)
        self.decoder1 = self.upconv_block(256, 64)        # Input: (batch_size, 256, 256, 256), Output: (batch_size, 64, 512, 512)

        # Output layer
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)  # Input: (batch_size, 128, 512, 512), Output: (batch_size, 1, 512, 512)

    def conv_block(self, in_channels, out_channels):
        # Returns a block with two convolutional layers
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        # Returns a transposed convolution for up-sampling
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)                             # (batch_size, 64, 512, 512)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))  # (batch_size, 128, 256, 256)
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))  # (batch_size, 256, 128, 128)
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))  # (batch_size, 512, 64, 64)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))  # (batch_size, 1024, 32, 32)

        # Decoder with concatenation from the contracting path (skip connections)
        dec4 = self.decoder4(bottleneck)  # (batch_size, 512, 64, 64)
        dec4 = self.center_crop_and_concat(enc4, dec4)  # (batch_size, 1024, 64, 64)
        
        dec3 = self.decoder3(dec4)  # (batch_size, 256, 128, 128)
        dec3 = self.center_crop_and_concat(enc3, dec3)  # (batch_size, 512, 128, 128)
        
        dec2 = self.decoder2(dec3)  # (batch_size, 128, 256, 256)
        dec2 = self.center_crop_and_concat(enc2, dec2)  # (batch_size, 256, 256, 256)
        
        dec1 = self.decoder1(dec2)  # (batch_size, 64, 512, 512)
        dec1 = self.center_crop_and_concat(enc1, dec1)  # (batch_size, 128, 512, 512)

        # Output layer
        mask = self.final_conv(dec1)  # (batch_size, 1, 512, 512)
        return mask

    def center_crop_and_concat(self, enc, dec):
        # Center crop the encoder feature maps to match the decoder feature maps
        enc_size = enc.size()[2:]  # (H, W)
        dec_size = dec.size()[2:]  # (H, W)

        # Calculate cropping dimensions
        crop_h = (enc_size[0] - dec_size[0]) // 2
        crop_w = (enc_size[1] - dec_size[1]) // 2

        # Center crop the encoder feature maps
        enc_cropped = enc[:, :, crop_h:crop_h + dec_size[0], crop_w:crop_w + dec_size[1]]
        return torch.cat((dec, enc_cropped), dim=1)  # Concatenate along channel axis
