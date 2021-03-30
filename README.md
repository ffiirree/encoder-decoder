# EncoderDecoder

What features are U networks need for image segmentation?

## EncoderDecoder Networks

- [ ] AE or VAE
- [ ] AE or VAE with shortcuts
- [x] `baseline:` vanilla UNet
- [x] UNet 2+ / UNet 3+
- [x] modified UNet 3+
- [x] UNet with two decoders / UNet with AE
- [ ] UNet with GAN

## with Carvana dataset

Net / Dice | vanilla | image reconstruction | object reconstruction  | border reconstruction
|--|:--:|:--:|:--:|:--:|
| AE | 97.99<br>98.02<br>98.03
| AE - 2 decoders | - |
| AE with shortcuts |
| AE with shortcuts - 2 decoders | - |
| UNet(`baseline`) | 98.87 <br> 98.65 <br> 98.86|
| UNet - 2 decoders | - |
| UNet 2+ |
| UNet 2+ - 2 decoders | - |
| UNet 3+ |
| UNet 3+ - 2 decoders | - |
| UNet 3+(modified) | 98.99<br>99.20<br>98.65
| UNet 3+(modified) - 2 decoders | - |

## with LITS 2017 dataset

Net| vanilla | image reconstruction | object reconstruction  | border reconstruction
|--|:--:|:--:|:--:|:--:|
| AE |
| AE - 2 decoders |
| AE with shortcuts |
| AE with shortcuts - 2 decoders |
| UNet(`baseline`) |
| UNet - 2 decoders |
| UNet 2+ |
| UNet 2+ - 2 decoders |
| UNet 3+ |
| UNet 3+ - 2 decoders |
| UNet 3+(modified) |
| UNet 3+(modified) - 2 decoders |
