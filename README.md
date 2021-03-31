# EncoderDecoder

What features are "U" / Encoder-Decoder networks needed for image segmentation?

## EncoderDecoder Networks

- [x] AutoEncoder
- [ ] AutoEncoder with shortcuts
- [x] AutoEncoder with two decoders
- [x] `baseline:` UNet
- [x] UNet with two decoders
- [x] UNet 2+ / (modified) UNet 3+
- [x] modified UNet 3+ with two decoders

## with Carvana dataset

Net / Dice | vanilla | image reconstruction | object reconstruction  | border reconstruction
|--|:--:|:--:|:--:|:--:|
| AE | 97.99<br>98.02<br>98.03
| AE - 2 decoders | - | 98.27 | 96.45
| AE with shortcuts |
| AE with shortcuts - 2 decoders | - |
| UNet(`baseline`) | 98.87 <br> 98.65 <br> 98.86|
| UNet - 2 decoders | - | 98.39<br>98.95
| UNet 2+ |
| UNet 2+ - 2 decoders | - |
| UNet 3+ |
| UNet 3+ - 2 decoders | - |
| UNet 3+(modified) | 
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
