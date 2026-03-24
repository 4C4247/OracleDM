<h1>OracleDM: Preserving Stroke Topology in Diffusion-Based Super-Resolution for Oracle Bone Inscriptions</h1>
<b>Authors</b>:
[Shibin Wang]<sup>1</sup>
| [Beigui Li]<sup>2</sup>
| [Xinyu Zhang]<sup>3</sup>
| [Yihang Li]<sup>3</sup>
| [Lu Yu]<sup>3</sup>

## 💬 At the beginning
Before starting to try, please read the following content

#  🛠️ How To Start？
## Dependencies and Installation

In the project, we simplified the naming of Oracle_DM and named it OBSR

- Pytorch >= 1.7.0
- CUDA >= 11.0
```
## Git clone this repository
git clone https://github.com/4C4247/Oracle_DM.git

cd OBSR

## Create new anaconda env
conda create -n OBSR python=3.8

conda activate OBSR
```
## Install the required packages
```
pip install -r requirements.txt
```
## 📢Regarding the function of each file
🌟The files in the IDM folder ensure the framework of our project.

🌟The files in the MoM folder and the utilities folder are the core of the stroke extraction module's operation.

🌟The files in the models folder are the main body of our model.

🌟The train folder contains the scripts required for two-stage model training.

🌟The checkpoins folder contains pretrained models.

## How to train?
 ❗ Just modify the configuration file as needed, and then run different training scripts to start training models with different functions.

If you need to train the stroke extraction model, select [train_stroke_extractor.py](Train/train_stroke_extractor.py) in the Train folder.
```
python train_stroke_extractor.py
```
If you need to train the diffusion model, select [train_two_dage_oracle_tsr.py](Train/train_two_stage_oracle_tsr.py) in the Train folder.
```
python train_two_stage_oracle_tsr.py
```

## How to inference?
 ❗ Enter the image to be tested, then configure and run the script in the [Inference](Inference/inference.py) folder according to the script content.
 ```
 python inference.py
 ```

## Pretrained Models
The pretrained model can be downloaded here[Google Drive](https://drive.google.com/drive/my-drive)

The two models correspond to the diffusion model and stroke extraction model respectively. Simply place them in the correct position to run normally!

## Dataset
The images used in our experiment are all from this [dataset](https://drive.google.com/drive/my-drive) and have been processed using different methods.

## 🔎 Overview of OBSR
![OBSR](Repo/entire.jpg)

<h4>Abstract</h4>
Oracle Bone Inscriptions (OBIs) constitute the core carrier of Chinese civilisation, yet their 
digital preservation is hindered by the low resolution and severe corrosion-induced complex degradation of existing rubbings. This degradation is particularly challenging due to
the intricate, straight, and broken stroke structures unique to OBI, which existing superresolution methods often misinterpret, leading to stroke adhesion and structural distortion.
To address these issues, we propose OracleDM, a novel diffusion-based super-resolution
model designed for high-fidelity OBI reconstruction, aiming to restore OBI images with
realistic style and clear strokes while ensuring the structural fidelity of inscriptions. Our
framework integrates a stroke extraction module and a mixture of modalities (MoM)
module to capture fine-grained stroke information directly from low-resolution images,
guiding the diffusion process without reliance on annotated font libraries. Evaluated on
our self-built OBI degradation dataset, OracleDM achieves state-of-the-art performance,
outperforming the leading DiffTSR method by 1.97 dB in PSNR and 0.027 in SSIM.
Here we show that our model not only significantly enhances resolution and visual
quality but also preserves critical stroke topology, ensuring authentic font details while
maintaining the stroke structure of OBI. This work introduces a new paradigm for the
digital protection of Oracle Bone Inscriptions by synergising diffusion models with the
prior knowledge of ancient Chinese characters, thereby facilitating deeper applications
of deep learning in the field of cultural heritage and humanities. 

## Visual effect preview
![OBSR](Repo/Preview.png)
