<h1>OracleDM: Preserving Stroke Topology in Diffusion-Based Super-Resolution for Oracle Bone Inscriptions</h1>

<b>Authors</b>:
[Shibin Wang]<sup>1</sup>
| [Beigui Li]<sup>1</sup>
| [Xinyu Zhang]<sup>1</sup>
| [Yihang Li]<sup>1</sup>
| [Lu Yu]<sup>1</sup>

<p><sup>1</sup> School of Computer and Information Engineering, Henan Normal University</p>

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch 2.1.1](https://img.shields.io/badge/PyTorch-2.1.1-ee4c2c)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()

## 💬 At the beginning

OracleDM reconstructs a clean, solid, glyph-correct **128×128** oracle bone character from a severely degraded low-resolution image whose **effective information is only about 16×16** pixels. The design priorities follow the epigraphic requirement, in this order:

> **glyph correctness (accurate strokes) > coherent, non-merged strokes > burr removal > sharpness.**

Unlike text-image super-resolution pipelines that rely on discrete semantic tokens, OracleDM uses a **continuous, stroke-level structural representation** and a **single image-diffusion backbone**.

## ✨ Highlights

- **Continuous four-channel condition pack** — skeleton, confidence, stroke width and appearance — extracted directly from the low-resolution image, with no fixed vocabulary.
- **Two continuous encoders** inject the geometric and appearance channels into one conditional UNet through ControlNet-style element-wise addition; there is no second diffusion process.
- **Three-stage curriculum**: structural formation → glyph-identity supervision with the Oracle-241 classifier → edge refinement.
- **Compound degradation modelling** plus **condition-aware post-processing** (speck removal, adaptive deburr, solidification) that cleans artifacts without truncating fine strokes.
- Compact backbone of about **20.9 million parameters**; fast inference with **25 DDIM steps**, guidance scale **1.0**, fixed seed **42**.

## 🛠️ Dependencies and installation

The code is tested with Python 3.10, PyTorch 2.1.1 and CUDA 12.1 (a CUDA-capable GPU is recommended; CPU inference works but is slow).

```bash
## Git clone this repository
git clone https://github.com/4C4247/Oracle_DM.git
cd Oracle_SkeletonDiff_v3

## Create a new anaconda environment
conda create -n OracleDM python=3.10 -y
conda activate OracleDM

## Install PyTorch (CUDA 12.1 build)
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

## Install the remaining packages
pip install numpy opencv-python scikit-image tqdm pillow flask matplotlib
```

## How to train

All learning-based models use the same training data and the same compound degradation; adjust the paths in the commands to your environment.

**Stage 1 — structural formation**

```bash
python train.py
```

**Stage 2 — glyph identity**

```bash
python train_stage2.py
```

**Stage 3 — edge refinement** (edge-quality and trajectory losses on the model's own DDIM path)

```bash
python train_stage2.py
```

## How to inference

**Command line** — final reconstruction only needs the Stage 3 checkpoint:

```bash
python generate.py
```
Google Drive (code and pretrained models): https://drive.google.com/drive/my-drive

## 📚 Datasets

Datasets:Google Drive (code and pretrained models): https://drive.google.com/drive/my-drive

## 🔎 Overview of OracleDM

![OracleDM architecture](paper/Paper_Final/figures/Final_framework.drawio.png)

## Abstract

Oracle bone inscriptions are among the earliest known Chinese writing systems, but the surviving photographic and rubbing records are often extremely low resolution: a nominal 64-pixel image may carry only about 16×16 pixels of effective information, with severe blur, noise, broken strokes and burrs. Recovering a clean, solid, glyph-correct 128×128 character from such input is a highly ill-posed super-resolution problem whose priorities are glyph correctness, coherent non-merged strokes, burr removal and sharpness. We propose OracleDM, a conditional diffusion model that replaces discrete text tokens with a continuous, stroke-level structural representation. A four-channel condition pack (skeleton, confidence, stroke width and appearance) is extracted from the low-resolution image and injected into a single image-diffusion backbone through two continuous encoders in a ControlNet-style additive manner rather than a coupled text–image dual-diffusion process. A three-stage curriculum, moving from structural formation to glyph-identity supervision with the Oracle-241 classifier and then to edge refinement, is combined with a compound degradation model and a condition-aware post-processing module. On a fixed 94-image subset with ground truth, OracleDM reaches 19.699 dB PSNR, 0.909 SSIM and 0.906 Dice, improving over the main diffusion baseline DiffTSR by 5.06 dB, 0.134 SSIM and 0.152 Dice while producing the fewest small holes. On a 96-image held-out set covering 26 unseen glyph classes, it reaches 14.378 dB and 0.829 Dice, and on 1440 real rubbings it yields solid, burr-free strokes. Ablations confirm the value of each condition channel, the three stages and the condition-aware post-processing.

## 📊 Visual effect preview

![In-domain qualitative comparison](paper/Paper_Final/figures/fig2_qualitative_indomain.png)

![Real rubbing reconstruction](paper/Paper_Final/figures/fig3_real_rubbings.png)

## Citation

If you find this work useful, please cite the following (a BibTeX entry will be added upon publication):

```
Wang, S., Li, B., Zhang, X., Li, Y., Yu, L.
OracleDM: Preserving Stroke Topology in Diffusion-Based Super-Resolution
for Oracle Bone Inscriptions. Heritage Science (Springer Nature).
```

## License

This project is released under the MIT License.
