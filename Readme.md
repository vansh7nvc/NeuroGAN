# NeuroGAN

NeuroGAN is a research-focused toolkit for training, evaluating, and using Generative Adversarial Networks (GANs) on neuroscience data (e.g., EEG, MEG, fMRI, calcium imaging, or neural spike trains). It provides configurable model architectures, training/evaluation scripts, and utilities to streamline experiments for data augmentation, simulation, and generative modeling research in computational neuroscience.

> NOTE: This README is a project-oriented template. Update dataset paths, config examples, model descriptions, and metrics to match the exact implementation and API in this repository.

## Table of Contents
- [Key features](#key-features)
- [Quick links](#quick-links)
- [Installation](#installation)
- [Requirements](#requirements)
- [Getting started (Quickstart)](#getting-started-quickstart)
  - [Prepare data](#prepare-data)
  - [Training](#training)
  - [Sampling / Generation](#sampling--generation)
  - [Evaluation](#evaluation)
- [Python API example](#python-api-example)
- [Supported models & architectures](#supported-models--architectures)
- [Datasets and preprocessing](#datasets-and-preprocessing)
- [Training details & configuration](#training-details--configuration)
- [Evaluation metrics](#evaluation-metrics)
- [Reproducibility & checkpoints](#reproducibility--checkpoints)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Key features
- Config-driven experiments (JSON / YAML configs)
- Modular model implementations (vanilla GAN, DCGAN-style, WGAN-GP, conditional variants)
- Data loaders and preprocessors for common neuroscience formats
- Checkpointing, logging, and TensorBoard support
- Scripted CLI for common workflows (train, eval, sample)
- Evaluation utilities for likelihood-free metrics and domain-specific measures

## Quick links
- Repository: https://github.com/vansh7nvc/NeuroGAN
- Example configs: `configs/` (edit and reuse)
- Training script: `scripts/train.py`
- Sampling script: `scripts/sample.py`
- Evaluation script: `scripts/evaluate.py`

(Adjust the above file paths if the files live somewhere else in the repo.)

## Installation

Recommended: create a Python virtual environment (venv or conda). Example using conda:

```bash
conda create -n neurogan python=3.10 -y
conda activate neurogan
pip install -r requirements.txt
# Or, for editable installation:
pip install -e .
```

If you don't have a requirements file, typical packages include:
- PyTorch (or TensorFlow, depending on implementation)
- numpy, scipy, scikit-learn
- matplotlib, seaborn
- pandas
- tqdm
- tensorboard (optional)
- h5py / nibabel (for neuroimaging formats)

Install CUDA-capable PyTorch if you plan to train on GPU:
https://pytorch.org/get-started/locally/

## Requirements

Minimum:
- Python 3.8+
- PyTorch 1.10+ (or TensorFlow 2.x if applicable)
- CUDA 11.x (optional, for GPU acceleration)

Refer to `requirements.txt` for an exact list (create one if missing).

## Getting started (Quickstart)

### Prepare data
1. Place your dataset in a directory such as `data/<dataset_name>/`.
2. For neuroimaging files (NIfTI), consider converting to tensors per subject/scan and normalizing.
3. For time series (EEG/MEG), segment into windows, baseline-correct, and standardize.

Example expected structure:
```
data/
  eeg_dataset/
    train/
      subject01.npy
      subject02.npy
      ...
    val/
      subjectXX.npy
```

Adjust preprocessing in `data/` utilities or write a custom DataLoader.

### Training

A typical training command uses a config file:

```bash
python scripts/train.py --config configs/eeg_wgan_gp.yaml --data-dir data/eeg_dataset --out-dir outputs/eeg_wgan
```

Important flags:
- --config : path to YAML/JSON configuration with hyperparameters
- --data-dir : root data directory
- --out-dir : where checkpoints / logs will be stored
- --device : `cuda` or `cpu`

Example minimal config fields:
```yaml
model:
  type: wgan-gp
  latent_dim: 128
  generator:
    channels: [256, 128, 64]
  discriminator:
    channels: [64, 128, 256]

training:
  batch_size: 64
  epochs: 200
  lr_g: 2e-4
  lr_d: 2e-4
  gp_lambda: 10
  n_critic: 5
```

### Sampling / Generation

To generate samples from a trained checkpoint:

```bash
python scripts/sample.py --ckpt outputs/eeg_wgan/checkpoint_latest.pt --num-samples 100 --out-dir outputs/eeg_wgan/samples
```

You can conditionally generate (if using cGAN) by passing labels or conditioning tensors.

### Evaluation

Evaluate generated samples against held-out real data:

```bash
python scripts/evaluate.py --real-dir data/eeg_dataset/val --fake-dir outputs/eeg_wgan/samples --metrics fid mmd
```

Supported metrics: FID, MMD, classification-based metrics, domain-specific signal measures (SNR, spectral power differences).

## Python API example

Use models and utilities directly in code:

```python
from neurogan.models import Generator, Discriminator
from neurogan.trainer import Trainer
from neurogan.data import NeuroDataset, get_dataloader

# dataset & dataloader
dataset = NeuroDataset("data/eeg_dataset/train")
loader = get_dataloader(dataset, batch_size=64, shuffle=True)

# models
G = Generator(latent_dim=128)
D = Discriminator()

# trainer
trainer = Trainer(G, D, loader, device="cuda")
trainer.train(epochs=200, out_dir="outputs/eeg_wgan")
```

(Adjust import paths to the actual package layout.)

## Supported models & architectures

NeuroGAN is designed to be modular. Example supported variants:
- Vanilla GAN
- DCGAN-style (convolutional)
- WGAN-GP (stable training with gradient penalty)
- Conditional GAN (cGAN)
- Style-based generators (experimental)
- Autoregressive / sequential variants for time-series data

Add or extend models in `neurogan/models/`.

## Datasets and preprocessing

Commonly used neuroscience datasets and formats:
- EEG/MEG: .npy, .mat, or FIF files (MNE or custom loaders)
- fMRI: NIfTI (.nii/.nii.gz) via nibabel
- Calcium imaging: HDF5, TIFF stacks
- Spiking data: sorted spike trains, binary rasters

Preprocessing suggestions:
- Bandpass filter for EEG/MEG (e.g., 1–40 Hz)
- Downsample to reasonable sampling rate
- Z-score or min-max normalization per channel
- Segment long recordings into windows of fixed length

Provide dataset-specific loader implementations and document expected shapes.

## Training details & configuration

- Save checkpoints periodically (e.g., every N epochs)
- Log losses and metrics to TensorBoard or CSV for easy visualization
- Use fixed noise vectors for visualizing training progression
- Seed RNGs for reproducibility (torch.manual_seed, numpy.random.seed)

Recommended hyperparameters (starting points):
- latent_dim: 64–256
- batch_size: 32–128 (depending on memory)
- lr: 1e-4 – 2e-4 (Adam)
- beta1: 0.0, beta2: 0.9 (for WGAN-GP)
- gp_lambda (WGAN-GP): 10

## Evaluation metrics

Standard and domain-specific metrics:
- Fréchet Inception Distance (FID) — if an Inception-like feature extractor exists for the domain
- Maximum Mean Discrepancy (MMD)
- Signal-to-Noise Ratio (SNR)
- Spectral/Power measures (e.g., relative band power: theta/alpha/beta)
- Classification accuracy using downstream task classifiers (i.e., are generated samples useful for augmentation?)
- Visual / qualitative inspection (plot signals, power spectral density, topographic maps)

Implement or adapt feature extractors for neuroscience modalities to compute FID-like scores.

## Reproducibility & checkpoints

- Checkpoint file format: model weights + optimizer states + training metadata (current epoch, config).
- To resume training:
```bash
python scripts/train.py --config configs/eeg_wgan_gp.yaml --resume outputs/eeg_wgan/checkpoint_latest.pt
```

- Include deterministic options and log RNG seeds in checkpoint metadata.

## Results

Document quantitative and qualitative results here:
- Example figure(s) showing real vs generated signals
- Quantitative tables for FID/MMD and domain-specific metrics
- Ablation study notes (latent dimension, model depth, loss variants)

(Replace this section with your experiment outputs and plots.)

## Contributing

Contributions are welcome! Typical workflows:
- Fork the repo and create a feature branch
- Add tests (if applicable) and ensure style consistency
- Create a clear PR describing the change and linking related issues

Suggested contribution topics:
- New dataset loaders
- Additional GAN variants or loss functions
- Evaluation metric implementations
- Tutorials and example notebooks

Please follow the repository's CONTRIBUTING.md (create one if absent) and code of conduct.

## Citation

If you use NeuroGAN in academic work, please cite this repository and any associated paper. Example BibTeX (edit when you have publication info):

```bibtex
@misc{NeuroGAN2025,
  author = {Your Name and Collaborators},
  title = {NeuroGAN: A GAN toolkit for neuroscience data},
  year = {2025},
  howpublished = {GitHub repository},
  note = {https://github.com/vansh7nvc/NeuroGAN}
}
```

## License

This project is released under the MIT License. See LICENSE for details. (Change license if needed.)

## Contact

Maintainer: vansh7nvc  
Repository: https://github.com/vansh7nvc/NeuroGAN

For questions, issues, or feature requests, please open an issue on the repository.
