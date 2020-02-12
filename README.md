# UNet: Cell segmentation with PyTorch


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

This model was trained from scratch with 150 images (with data augmentation)This score could be improved with more training, data augmentation, fine tuning, playing with CRF post-processing, and applying more weights on the edges of the masks.



## Usage
**Note : Use Python 3**

The input images and target masks should be in the `data/cells/scans` and `data/cells/labels` folders respectively.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
