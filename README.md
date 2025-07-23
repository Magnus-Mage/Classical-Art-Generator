# CycleGAN Image Translation

A PyTorch implementation of CycleGAN for unpaired image-to-image translation, based on the original work by Jun-Yan Zhu et al.

## Overview

CycleGAN enables image translation between two domains without requiring paired training examples. This implementation focuses specifically on the CycleGAN architecture, providing tools for training and testing unpaired image translation models.

## Key Features

- Unpaired image-to-image translation using cycle-consistent adversarial networks
- Support for various image domains (horses↔zebras, photos↔paintings, etc.)
- Pre-trained models available for immediate use
- Comprehensive training and testing pipeline
- Integration with Visdom for training visualization
- W&B (Weights & Biases) logging support

## Requirements

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- PyTorch 0.4+

## Installation

1. Clone this repository:
```bash
git clone git@github.com:Magnus-Mage/Classical-Art-Generator.git
cd Classical-Art-Generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using Conda:
```bash
conda env create -f environment.yml
```

## Quick Start

### Using Pre-trained Models

1. Download a pre-trained model (e.g., horse2zebra):
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```

2. Download the corresponding dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

3. Generate results:
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```

### Training Your Own Model

1. Download a dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```

2. Start Visdom server for visualization:
```bash
python -m visdom.server
```
Visit http://localhost:8097 to view training progress.

3. Train the model:
```bash
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

4. Test the trained model:
```bash
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

## Available Datasets

The repository includes scripts to download various datasets:
- horse2zebra
- maps (aerial ↔ map)
- cityscapes
- facades
- apple2orange
- summer2winter_yosemite
- monet2photo
- cezanne2photo
- ukiyoe2photo
- vangogh2photo

## Model Architecture

CycleGAN consists of:
- Two generator networks (G: X→Y, F: Y→X)
- Two discriminator networks (D_X, D_Y)
- Cycle consistency loss to ensure F(G(x)) ≈ x and G(F(y)) ≈ y
- Adversarial losses for realistic image generation

## Training Tips

- Monitor training through Visdom at http://localhost:8097
- Use `--use_wandb` flag for W&B logging
- Adjust `--lambda_A` and `--lambda_B` for cycle consistency weight
- Consider `--pool_size` for discriminator update frequency
- Use `--continue_train` to resume training from checkpoints

## Results

Training results are saved to `./checkpoints/{name}/web/index.html`
Test results are saved to `./results/{name}/test_latest/index.html`

## File Structure

```
├── datasets/           # Dataset download scripts
├── models/            # Model definitions
├── data/              # Dataset loading utilities
├── scripts/           # Training/testing scripts
├── checkpoints/       # Saved models and training logs
├── results/           # Test output images
└── docs/              # Documentation
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```

## Contributors

- [Magnus-Mage](https://github.com/Magnus-Mage)
- [DhruvKikan](https://github.com/DhruvKikan)

## Acknowledgments

This implementation is based on the original CycleGAN work by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. The code structure is inspired by pytorch-DCGAN.

## Related Projects

- [Original CycleGAN Paper](https://arxiv.org/pdf/1703.10593.pdf)
- [CycleGAN Project Page](https://junyanz.github.io/CycleGAN/)
- [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation)

## License

This project follows the same license as the original CycleGAN implementation.
