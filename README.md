# Saliency-detection-in-360-video
Saliency Detection in 360$^\circ$ Videos (ECCV2018)

This project hosts the dataset and code for our ECCV 2018 paper

- Ziheng Zhang, Yanyu Xu, Jingyi Yu, and Shenghua Gao, Saliency Detection in 360$^\circ$ Videos

This paper presents a novel spherical convolutional neural network based scheme for saliency detection for $360^\circ$ videos. Specifically, in our spherical convolution neural network definition, kernel is defined on a spherical crown, and the convolution involves the rotation of the kernel along the sphere. Considering that the $360^\circ$ videos are usually stored with equirectangular panorama, we propose to implement the spherical convolution on panorama by stretching and rotating the kernel based on the location of patch to be convolved. Compared with existing spherical convolution, our definition has the parameter sharing property, which would greatly reduce the parameters to be learned. We further take the temporal coherence of the viewing process into consideration, and propose a sequential saliency detection by leveraging a spherical U-Net. To validate our approach, we construct a large-scale $360^\circ$ videos saliency detection benchmark that consists of 104 $360^\circ$ videos viewed by 20+ human subjects. Comprehensive experiments validate the effectiveness of our spherical U-net for $360^\circ$ video saliency detection.

## 360 Video Saliency Dataset

To be continued.
