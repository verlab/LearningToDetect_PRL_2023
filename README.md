# Improving the matching of deformable objects by learning to detect keypoints
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DALF_CVPR_2023/blob/main/notebooks/registration_with_DALF.ipynb)-->
### [ArXiv](https://arxiv.org/abs/2309.00434)
<!--- | [Project Page](https://www.verlab.dcc.ufmg.br/descriptors/dalf_cvpr23/)>
<!--<img align="center" src='./figs/example_reg.png' align="center"/>

<div align="center">
<img src='./figs/hard_deform.gif' align="center" width="640"/> <br>
DALF registration with challenging deformation + illumination + rotation transformations. <br><br>
</div>

**TL;DR**: A joint image keypoint detector and descriptor for handling non-rigid deformation. Also works great under large rotations. -->

## Table of Contents
- [Introduction](#introduction) <!--img align="right" src='./figs/arch.png' width=360 /-->
- [Requirements](#requirements)
- [Installation](#installation)
- [Inference](#inference)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Introduction
This repository contains the official implementation of the paper: *[Improving the matching of deformable objects by learning to detect keypoints](https://arxiv.org/abs/2309.00434)* published in Elsevier's Pattern Recognition Letters 2023.

**Abstract**: We propose a novel learned keypoint detection method to increase the number of correct matches for the task of non-rigid image correspondence. By leveraging true correspondences acquired by matching annotated image pairs with a specified descriptor extractor, we train an end-to-end convolutional neural network (CNN) to find keypoint locations that are more appropriate to the considered descriptor. Experiments demonstrate that our method enhances the Mean Matching Accuracy of numerous descriptors when used in conjunction with our detection method, while outperforming the state-of-the-art keypoint detectors on real images of non-rigid objects by 20 p.p. We also apply our method on the complex real-world task of object retrieval where our detector performs on par with the finest keypoint detectors currently available for this task

<!--**Overview of DALF achitecture**
Our architecture jointly optimizes non-rigid keypoint detection and description, and explicitly models local deformations for descriptor extraction during training. An hourglass CNN computes a dense heat map providing specialized keypoints that are used by the Warper Net to extract deformation-aware matches. A feature fusion layer balances the trade-off between invariance and distinctiveness in the final descriptors. DALF network is used to produce a detection heatmap and a set of local features for each image. In the detector path, the heatmaps are optimized via the REINFORCE algorithm considering keypoint repeatability under deformations. In the descriptor path, feature space is learned via the hard triplet loss. A siamese setup using image pairs is employed to optimize the network.

<img align="center" src='./figs/training.png' align="center" width=860 /-->


## Requirements
- [conda](https://docs.conda.io/en/latest/miniconda.html) for automatic installation;

## Installation
Clone the repository, and build a fresh conda environment for this detector:
```bash
git clone https://github.com/verlab/PRL2023.git
cd PRL2023
conda env create -f env.yml -n prl_env
conda activate prl_env
```

<!--### Manual installation
In case you just want to manually install the dependencies, first install [pytorch (>=1.12.0)](https://pytorch.org/get-started/previous-versions/) and then the rest of depencencies:
```bash
#For GPU (please check your CUDA version)
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
#CPU only
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu

pip install --user numpy scipy opencv-contrib-python kornia
```

## Usage

For your convenience, we provide ready to use notebooks for some tasks.

|            **Description**     |  **Notebook**                     |
|--------------------------------|-------------------------------|
| Matching example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DALF_CVPR_2023/blob/main/notebooks/registration_with_DALF.ipynb) |
| Register a video of deforming object (as shown in the GIF)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DALF_CVPR_2023/blob/main/notebooks/video_registration_with_DALF.ipynb) |
| Download data and train from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DALF_CVPR_2023/blob/main/notebooks/train_DALF_from_scratch.ipynb) |-->


### Inference
To run Our detector on an image, use this [script](./run_detector.py) in the root folder:
```bash
python run_detector.py
```

<!--### Training
DALF can be trained in a self-supervised manner with synthetic warps (see [augmentation.py](modules/dataset/augmentation.py)), i.e., one can use a folder with random images for training. In our experiments, we used the raw images (without any annotation) of 1DSfM datasets which can be found in [this link](https://www.cs.cornell.edu/projects/1dsfm/).
To train DALF from scratch on a set of arbitrary images with default parameters, run the following command:
```bash
python3 train.py
```
To train the model, we recommend a machine with a GPU with at least 10 GB memory, and 16 GB of RAM. You can attempt to reduce the batch size and increase the number of gradient accumulations accordingly, to train in a GPU with less than 10 GB.
We provide a Colab to demonstrate how to train DALF from scratch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/DALF_CVPR_2023/blob/main/notebooks/train_DALF_from_scratch.ipynb). While it is possible to train the model on Colab, it should take more than 48 hours of GPU usage.

### Evaluation
We follow the same protocol and benchmark evaluation of [DEAL](https://github.com/verlab/DEAL_NeurIPS_2021#vi---evaluation). You will need to [download the non-rigid evaluation benchmark files](https://www.verlab.dcc.ufmg.br/descriptors/#datasets). Then, run the [evaluation script](eval/eval_nonrigid.sh):
```bash
sh ./eval/eval_nonrigid.sh
```
Please update the variables ``PATH_IMGS`` and ``PATH_TPS`` to point to your downloaded benchmark files before running the evaluation script!

## Applications

The image retrieval and non-rigid surface registration used in the paper will be released very soon in a new repository focused on application tasks involving local features. Stay tuned!

The video below show the non-rigid 3D surface registration results from the paper:
<p align="center">
  <a href="https://www.youtube.com/watch?v=7-wDqrhn33Y"><img src="https://img.youtube.com/vi/7-wDqrhn33Y/0.jpg" alt="Non-rigid 3D registration visual results"></a>
</p-->

## Citation
If you find this code useful for your research, please cite the paper:

```bibtex
@article{CADAR2023,
 title = {Improving the matching of deformable objects by learning to detect keypoints},
 journal = {Pattern Recognition Letters},
 year = {2023},
 issn = {0167-8655},
 doi = {https://doi.org/10.1016/j.patrec.2023.08.012},
 url = {https://www.sciencedirect.com/science/article/pii/S0167865523002325},
 author = {Felipe Cadar and Welerson Melo and Vaishnavi Kanagasabapathi and Guilherme Potje and Renato Martins and Erickson R. Nascimento}
}
```

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements
- We would like to thank CAPES, CNPq, FAPEMIG, Google, and Conseil Régional BFC for funding different parts of this work and NVIDIA for the donation of a Titan XP GPU used for this study.
- This work was also granted access to the HPC resources of IDRIS under the project 2021-AD011013154.

**VeRLab:** Laboratory of Computer Vison and Robotics https://www.verlab.dcc.ufmg.br
<br>
<img align="left" width="auto" height="50" src="./assets/ufmg.png">
<img align="right" width="auto" height="50" src="./assets/verlab.png">
<br/>
