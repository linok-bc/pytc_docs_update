<a href="https://github.com/zudi-lin/pytorch_connectomics">
<img src="./.github/logo_fullname.png" width="450"></a>

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8-2BAF2B.svg" /></a>
    <a href= "https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href= "https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html">
      <img src="https://img.shields.io/badge/Documentation-Latest-2BAF2B.svg" /></a>
    <a href= "https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html">
      <img src="https://img.shields.io/badge/Slack-Join-CC8899.svg" /></a>
</p>

<hr/>

## Introduction

The field of *connectomics* aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses. Recent advances in electronic microscopy (EM) have enabled the collection of a large number of image stacks at nanometer resolution, but the annotation requires expertise and is super time-consuming. Here we provide a deep learning framework powered by [PyTorch](https://pytorch.org/) for automatic and semi-automatic semantic and instance segmentation in connectomics, which is called **PyTorch Connectomics** (PyTC). This repository is mainly maintained by the Visual Computing Group ([VCG](https://vcg.seas.harvard.edu)) at Harvard University.

*PyTorch Connectomics is currently under active development!*

## Key Features

- Multi-task, Active and Semi-supervised Learning
- Distributed and Mixed-precision Training
- Scalability for Handling Large Datasets

If you want new features that are relatively easy to implement (*e.g.*, loss functions, models), please open a feature requirement discussion in issues or implement by yourself and submit a pull request. For other features that requires substantial amount of design and coding, please contact the [author](https://github.com/zudi-lin) directly.

## Environment

The code is developed and tested under the following configurations.

- Hardware: 1-8 Nvidia GPUs with at least 12G GPU memory (change ```SYSTEM.NUM_GPU``` accordingly based on the configuration of your machine)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=11.0, Python>=3.8, PyTorch>=1.8.0, YACS>=0.1.8***

## Installation

Create a new conda environment:

```shell
conda create -n py3_torch python=3.8
source activate py3_torch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Please note that this package is mainly developed on the Harvard [FASRC](https://www.rc.fas.harvard.edu) cluster. More information about GPU computing on the FASRC cluster can be found [here](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

Download and install the package (editable):

```shell
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install --upgrade pip
pip install -r requirements.txt
pip install --editable .
```

For more information and frequently asked questions about installation, please check the [installation guide](https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/installation.html).

## Notes

### Data Augmentation

We provide a data augmentation interface several different kinds of commonly used augmentation method for EM images. The interface is pure-python, and operate on and output only numpy arrays, so it can be easily incorporated into any kinds of python-based deep learning frameworks (e.g., TensorFlow). For more details about the design of the data augmentation module, please check the [documentation](https://zudi-lin.github.io/pytorch_connectomics/build/html/modules/data.html#module-connectomics.data.augmentation).

### YACS Configuration

We use the *Yet Another Configuration System* ([YACS](https://github.com/rbgirshick/yacs)) library to manage the settings and hyperparameters in model training and inference. The configuration files for tutorial examples can be found [here](https://github.com/zudi-lin/pytorch_connectomics/tree/master/configs). All available configuration options can be found at [```connectomics/config/config.py```](https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/config/config.py). Please note that the default value of several options is ```None```, which is only supported after YACS v0.1.8.

### Segmentation Models

We provide several encoder-decoder architectures, which are customized 3D UNet and Feature Pyramid Network (FPN) models with various blocks and backbones. Those models can be applied for both semantic segmentation and bottom-up instance segmentation of 3D image stacks. Those models can also be constructed specifically for isotropic
and anisotropic datasets. Please check the [documentation](https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html) for more details.

## Acknowledgement

This project is built upon numerous previous projects. Especially, we'd like to thank the contributors of the following github repositories:

- [pyGreenTea](https://github.com/naibaf7/PyGreentea): HHMI Janelia FlyEM Team
- [DataProvider](https://github.com/torms3/DataProvider): Princeton SeungLab
- [Detectron2](https://github.com/facebookresearch/detectron2): Facebook AI Reserach

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Citation

If you find PyTorch Connectomics (PyTC) useful in your research, please cite:

```bibtex
@misc{lin2019pytorchconnectomics,
  author =       {Zudi Lin and Donglai Wei},
  title =        {PyTorch Connectomics},
  howpublished = {\url{https://github.com/zudi-lin/pytorch_connectomics}},
  year =         {2019}
}
```
