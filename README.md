# Test-Time Stain Adaptation with Diffusion Models for Histopathology Image Classification
> [**Test-Time Stain Adaptation with Diffusion Models for Histopathology Image Classification**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05175.pdf)  
> Cheng-Chang Tsai, Yuan-Chih Chen, and Chun-Shien Lu.  
> In *European Conference on Computer Vision*. 2024.

## Abstract
Stain shifts are prevalent in histopathology images, and typically dealt with by normalization or augmentation. Considering trainingtime methods are limited in dealing with unseen stains, we propose a test-time tain adaptation method (TT-SaD) with diffusion models that achieves stain adaptation by solving a nonlinear inverse problem during testing. TT-SaD is promising in that it only needs a single domain for training but can adapt well from other domains during testing, preventing models from retraining whenever there are new data available. For tumor classification, stain adaptation by TT-SaD outperforms stateof-the-art diffusion model-based test-time methods. Moreover, TT-SaD beats training-time methods when testing on data that are inaccessible during training. To our knowledge, the study of stain adaptation in diffusion model during testing time is relatively unexplored.

## Introduction
This repo is based on [guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for **stain adaptation**.

## Installation
Clone this repo and navigate to it in your terminal. Then run:
```
pip install -e .
```
Additionally, run
```
git clone https://github.com/aetherAI/stain-mixup.git
```
and follow the instructions there for downloading [spams](http://thoth.inrialpes.fr/people/mairal/spams/).

## Usage
### Stain Adaptation
```
bash image_sample.sh
```
Before running the above command, one should:
- train a diffusion model on the source data,
- calculate the domain center.

### Data
The datasets used in this work can be downloaded from:
- [CAMELYON17](https://camelyon17.grand-challenge.org/)
- [MITOS-ATYPIA-14](https://mitos-atypia-14.grand-challenge.org/)

### Diffusion Models
The diffusion model of **TT-SaD** is trained using [improved-diffusion](https://github.com/openai/improved-diffusion).

### Domain Center
The domain center is calculated in the following three steps:
1. Using `get_stain_matrix()` in [stain-mixup](https://github.com/aetherAI/stain-mixup/blob/main/stain_mixup/utils.py#L31C5-L31C21) to extract all stain matrices from the source data.
2. Calculate the mean of all stain matrices.
3. Find the stain matrix closest to the mean.

## Citation
If you find this repo helpful, please cite the following work without hesitation:
```
@inproceedings{tsai2024test,
  title={Test-Time Stain Adaptation with Diffusion Models for Histopathology Image Classification},
  author={Tsai, Cheng-Chang and Chen, Yuan-Chih and Lu, Chun-Shien},
  booktitle={European Conference on Computer Vision},
  pages={257--275},
  year={2024},
  organization={Springer}
}
```
