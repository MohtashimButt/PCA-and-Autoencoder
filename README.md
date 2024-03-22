# PCA and Autoencoder implementation for image denoinsing, image reconstruction, and image segmentation

## `NB1.ipynb`
This notebook contains work done mostly on CIFAR-10 and MNIST dataset. 

### Part-1
This part is just a *CNN-based image classification* on CIFAR-10 dataset. The CNN architecutre contains the residual blocks with 47,650 of total parameters ran on 20 epochs and gave the accuracy of 74%. The CNN architecture is as follows:

<center>![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/architecture_CNN.png)</center>

The training loss curve and test accuracy curve are given as follows:
<center>![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/CNN_out.png)</center>

### Part-2
This part applies Principle Component Analysis on MNIST dataset to reconstruct the images as well as denoise the noisy image. The architecture followed for this process as as follows:
<center>![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_arch.png)</center>

The training and testing loss curves are given as follows:
<center>![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_out.png)</center>

The image reconstruction and image denoising results are given as follows:
| Image Reconstruction | Image Denoising |
| -------------------- | --------------- |
| ![Image Reconstruction](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_recon.png) | ![Image Denoising](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_denoise.png) |
