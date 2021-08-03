# Single Image Pretraining of Visual Representations

As shown in the paper

**A critical analysis of self-supervision, or what we can learn from a single image**, Asano et al. ICLR 2020


![Example images from our dataset](images/_example.png?raw=true "Example Images")

## Why?
Self-supervised representation learning has made enormous strides in recent years.
In this paper we show that a large part why self-supervised learning works are the augmentations.
We show this by pretraining various SSL methods on a dataset generated solely from augmenting a single source image
and find that various methods still pretrain quite well and even yield representations as strong as using the whole dataset for the early layers of networks.


## Abstract
We look critically at popular self-supervision techniques for learning deep convolutional neural networks without manual labels. We show that three different and representative methods, BiGAN, RotNet and DeepCluster, can learn the first few layers of a convolutional network from a single image as well as using millions of images and manual labels, provided that strong data augmentation is used. However, for deeper layers the gap with manual supervision cannot be closed even if millions of unlabelled images are used for training. We conclude that: (1) the weights of the early layers of deep networks contain limited information about the statistics of natural images, that (2) such low-level statistics can be learned through self-supervision just as well as through strong supervision, and that (3) the low-level statistics can be captured via synthetic transformations instead of using a large image dataset.

## Usage
Here we provide the code for generating a dataset from using just a single source image.
Since the publication, I have slightly modified the dataset generation script to make it easier to use.
Dependencies: `torch, torchvision, joblib, PIL, numpy`, any recent version should do.

Run like this:
```
python make_dataset_single.py --imgpath images/ameyoko.jpg --targetpath ./out/ameyoko_dataset
```

Here is the full description of the usage:
```
usage: make_dataset_single.py [-h] [--img_size IMG_SIZE]
                              [--batch_size BATCH_SIZE] [--num_imgs NUM_IMGS]
                              [--threads THREADS] [--vflip] [--deg DEG]
                              [--shear SHEAR] [--cropfirst]
                              [--initcrop INITCROP] [--scale SCALE SCALE]
                              [--randinterp] [--imgpath IMGPATH] [--debug]
                              [--targetpath TARGETPATH]

Single Image Pretraining, Asano et al. 2020

optional arguments:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE
  --batch_size BATCH_SIZE
  --num_imgs NUM_IMGS   number of images to be generated
  --threads THREADS     how many CPU threads to use for generation
  --vflip               use vflip?
  --deg DEG             max rot angle
  --shear SHEAR         max shear angle
  --cropfirst           usage of initial crop to not focus too much on center
  --initcrop INITCROP   initial crop size relative to image
  --scale SCALE SCALE   data augmentation inverse scale
  --randinterp          For RR crops: use random interpolation method or just bicubic?
  --imgpath IMGPATH
  --debug
  --targetpath TARGETPATH
```

## Reference
If you find this code/idea useful, please consider citing our paper:
```
@inproceedings{asano2020a,
title={A critical analysis of self-supervision, or what we can learn from a single image},
author={Asano, Yuki M. and Rupprecht, Christian and Vedaldi, Andrea},
booktitle={International Conference on Learning Representations (ICLR)},
year={2020},
}
```
