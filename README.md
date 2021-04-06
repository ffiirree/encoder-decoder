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
| AE |98.80<br>98.52<br>98.25<br>98.15
| AE - 2 decoders | - | 98.11<br>98.83<br>98.27<br>98.46 | 98.64<br>98.75<br>98.57<br>98.13
| AE with shortcuts |
| AE with shortcuts - 2 decoders | - |
| UNet(`baseline`) |98.71<br>98.62<br>98.87<br>98.56
| UNet - 2 decoders | - | 98.78<br>98.81<br>98.65<br>98.92 | 99.08<br>98.29<br>98.08<br>98.98
| UNet 2+ |
| UNet 2+ - 2 decoders | - |
| UNet 3+ | 98.79<br>99.10<br>98.82<br>98.84 |
| UNet 3+ - 2 decoders | - | 98.89<br>99.12<br>99.02<br>99.23 |98.84<br>98.74<br>98.83<br>99.13

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
