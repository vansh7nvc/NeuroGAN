# NeuroGAN

NeuroGAN is a research-oriented PyTorch implementation of generative adversarial networks (GANs) for image synthesis and experimentation. The repository provides modular, extensible code for training, sampling, and evaluating popular GAN variants, plus utilities for datasets, logging, and checkpointing.

Key goals:
- Easy experimentation with architectures and loss functions
- Reproducible training and evaluation
- Clear examples for sampling and model evaluation (FID/IS)

--------------------------------------------------------------------------------
Table of contents
- Features
- Repository structure
- Getting started
- Training
- Sampling / Inference
- Evaluation
- Configuration & hyperparameters
- Checkpoints & resuming
- Common workflows
- Tips & troubleshooting
- Contributing
- License & citation

--------------------------------------------------------------------------------
Features
- Modular model code (Generator, Discriminator) with clear interfaces
- Support for common GAN losses (vanilla, WGAN-GP, hinge)
- Data loaders for common image datasets (CIFAR-10, CelebA, custom)
- Scripted training loop with logging (TensorBoard) and checkpointing
- Utilities for computing FID (and optionally IS)
- Example configs for baseline experiments

--------------------------------------------------------------------------------
Repository structure
- configs/           — example config files or hyperparameter presets
- datasets/          — dataset wrappers and preprocessing utilities
- models/            — Generator, Discriminator, and model helpers
- trainers/          — training loops and scheduler utilities
- utils/             — logging, checkpointing, metrics (FID), seeding
- scripts/
  - train.py         — entrypoint for training experiments
  - sample.py        — generate images from checkpoints
  - evaluate.py      — compute FID/IS against a dataset
- docs/              — notes, experiments, and usage examples
- tests/             — unit / integration tests (if present)

--------------------------------------------------------------------------------
Getting started (quick)
Prerequisites
- Python 3.8+
- PyTorch (installed with CUDA support if using GPU)
- CUDA-enabled GPU recommended for training

Quick install
1. Clone the repo:
   git clone https://github.com/vansh7nvc/NeuroGAN.git
   cd NeuroGAN

2. (Recommended) Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt
   # If requirements.txt is not present, install PyTorch and common packages:
   pip install torch torchvision tensorboard numpy matplotlib tqdm pillow

4. Download / prepare datasets:
   - For example, to use CIFAR-10 the dataset loader will download automatically
     (or provide a path to your dataset in the config).

--------------------------------------------------------------------------------
Training
Basic training command:
python scripts/train.py --config configs/cifar10_base.yaml --outdir runs/exp1

Common CLI options:
- --config : path to YAML/JSON config with hyperparameters
- --data   : dataset name or path
- --batch-size
- --epochs
- --gpus

What the trainer does:
- Loads model, optimizer, and schedulers from config
- Prepares dataloader with augmentations / normalization
- Runs training loop with per-epoch evaluation & checkpointing
- Logs metrics to TensorBoard and a CSV file

Example config keys (configs/cifar10_base.yaml):
- model:
    generator:
      z_dim: 128
      hidden: 256
    discriminator:
      hidden: 256
- optim:
    g_lr: 2e-4
    d_lr: 2e-4
- loss: "hinge"      # options: vanilla, wgan-gp, hinge
- data:
    name: "cifar10"
    image_size: 32
- training:
    batch_size: 64
    epochs: 200
    grad_penalty_weight: 10.0   # for WGAN-GP

--------------------------------------------------------------------------------
Sampling / Inference
Generate samples from a checkpoint:
python scripts/sample.py --checkpoint path/to/checkpoint.pt --num-samples 64 --outdir samples/exp1

Options include:
- sample fixed seeds (for reproducibility)
- sample with truncation (if using style-based architectures)
- save grid or individual images

--------------------------------------------------------------------------------
Evaluation
Compute FID (Fréchet Inception Distance) and optionally Inception Score:
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --dataset cifar10 --num-samples 5000

Notes:
- FID requires a reference set of real images; scripts either use the test split or precomputed statistics.
- For accurate FID, generate at least several thousand samples.
- If you need exact reproductions of literature scores, follow dataset preprocessing and image resizing used in the original paper.

--------------------------------------------------------------------------------
Configuration & hyperparameters
- All training-relevant hyperparameters should be set in config files to ensure reproducibility.
- Use deterministic seeds (see utils/seed.py) when comparing experiment runs.
- Recommended defaults (good starting points):
  - z_dim: 128
  - batch_size: 64 (increase with GPU memory)
  - g_lr/d_lr: 2e-4
  - optimizer: Adam(betas=(0.5, 0.999))
  - training epochs: 100-200 (dataset dependent)

--------------------------------------------------------------------------------
Checkpoints & resuming
- Checkpoints store model state_dicts, optimizer states, epoch, and RNG seeds.
- By default checkpoints are saved to runs/{experiment}/checkpoints.
- Resume training:
  python scripts/train.py --config ... --resume runs/exp1/checkpoints/ckpt_last.pt

--------------------------------------------------------------------------------
Common workflows
- Quick prototyping: use low resolution (32x32), small batch sizes and fewer epochs.
- Ablation: change only one config parameter at a time and log experiment metadata.
- Large runs: use a dedicated machine with multiple GPUs and set distributed options in the config.

--------------------------------------------------------------------------------
Tips & troubleshooting
- Out-of-memory: lower batch size, use gradient accumulation, or use a smaller model.
- Training unstable: try switching to hinge loss, add spectral normalization, tune learning rates, or increase discriminator updates per generator update.
- Low-quality samples: check dataset preprocessing, normalization ([-1,1] vs [0,1]), and seed consistency.

--------------------------------------------------------------------------------
Contributing
Contributions are welcome. Suggested steps:
1. Open an issue to discuss major changes before implementing.
2. Fork the repo and create a feature branch.
3. Keep commits small and focused. Add tests for new functionality where appropriate.
4. Open a pull request describing changes and motivation.

--------------------------------------------------------------------------------
License
This project is released under the MIT License. See LICENSE for details.

--------------------------------------------------------------------------------
Citing NeuroGAN
If you use this code in published research, please cite the repository and any relevant papers associated with architectures or metrics you used.

--------------------------------------------------------------------------------
Acknowledgements
This implementation draws on common GAN research and open-source implementations. See HISTORY.md or docs/ for more detail on references and related work.

--------------------------------------------------------------------------------
Contact
Maintainer: vansh7nvc
GitHub: https://github.com/vansh7nvc/NeuroGAN
