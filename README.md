# Cavity-nnUNet-PyTorch
This repository contains simplified codes for "Mycobacterial cavity on chest CT: clinical implications and deep learning-based automatic detection with quantification", which was submitted to Quantitative Imaging in Medicine and Surgery. The purpose is to automatically detect and volumetrically quantify mycobacterial cavity volume on CT images.

Paper
===============
* Mycobacterial cavity on chest CT: clinical implications and deep learning-based automatic detection with quantification (submitted to Quantitative Imaging in Medicine and Surgery.)

Implementation
===============
A PyTorch implementation of Lung Cavity Segmentation.
We implemented this code based on the [nnU-Net code](https://github.com/MIC-DKFZ/nnUNet).

During the pre-processing stage, we applied filtration process to aid the cavity segmentation. Hounsfield unit (HU) values outside the lung area were filtered to -1024. The lung parenchyma was segmented by a previously developed deep neural network which automatically extracts lung parenchyma from CT images.

* Our Experiment Environment:
  * OS : Ubuntu
  * Python 3.8.10
  * PyTorch 1.11.0


Data
===============
Data are available from the authors upon reasonable request.

Citations
===============
```
To be announced after publication
```
