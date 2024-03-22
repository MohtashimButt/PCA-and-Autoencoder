# PCA and Autoencoder implementation for image denoising, image reconstruction, and image segmentation

## `NB1.ipynb`
This notebook contains work done mostly on CIFAR-10 and MNIST dataset.

### Part-1
This part is just a *CNN-based image classification* on CIFAR-10 dataset. The CNN architecture contains the residual blocks with 47,650 of total parameters ran on 20 epochs and gave the accuracy of 74%. The CNN architecture is as follows:  
![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/architecture_CNN.png)

The training loss curve and test accuracy curve are given as follows:  
<center> <img src="https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/CNN_out.png" alt="cnn_arch"> </center>
<!-- ![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/CNN_out.png) -->

### Part-2
This part applies Principle Component Analysis on MNIST dataset to reconstruct the images as well as denoise the noisy image. The architecture followed for this process as as follows (with 223,391 parameters):  
![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_arch.png)

The training and testing loss curves are given as follows:  
![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_out.png)

The image reconstruction and image denoising results are given as follows:  

| Image Reconstruction | Image Denoising |
| -------------------- | --------------- |
| <center> <img src="https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_recon.png" alt="Image Reconstruction"> </center> | <center> <img src="https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/PCA_denoise.png" alt="Image Denoising"> </center> |

> The results (as you can see) are not that promising. To resolve this, we'll use a convulotion Encoder-Decoder model (aka Autoender) instead of just linear PCA

### Part-3
This part applies Autoencoder on MNIST dataset to reconstruct the images as well as denoise the noisy image. The architecture followed for this process as as follows (with only 28,129 parameters):  
![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/AE_arch.png)

The training and testing loss curves are given as follows: 
![cnn_arch](https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/AE_out.png)

The image reconstruction and image denoising results are given as follows:  

| Image Reconstruction | Image Denoising |
| -------------------- | --------------- |
| <center> <img src="https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/AE_recon.png" alt="Image Reconstruction"> </center> | <center> <img src="https://github.com/MohtashimButt/PCA-and-Autoencoder/blob/master/Assets/AE_denoise.png" alt="Image Denoising"> </center> |

> With conparatively very less parameters, Autoencoders still outperformed PCA. 