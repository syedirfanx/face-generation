# CelebA Face Generation using Deep Convolutional GAN (DCGAN)

BSc Academic Project - CSE 465 Deep Learning, 2019

A PyTorch implementation of a Deep Convolutional Generative Adversarial Network trained on the CelebA dataset to synthesize photorealistic human face images.

---

## Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human face images. The model is trained on the [CelebA (Large-scale CelebFaces Attributes)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, which contains over 200,000 celebrity face images.

A GAN consists of two competing neural networks:
- **Generator (G):** Learns to produce fake images from random noise vectors that are indistinguishable from real images.
- **Discriminator (D):** Learns to distinguish between real images from the dataset and fake images produced by the Generator.

The two networks are trained simultaneously in a minimax game until equilibrium is reached and the Generator can produce convincing face images.

---

## Project Structure

```
face-generation/
|
|-- dcgan/
|   |-- DCGAN.ipynb
|
|-- problemunittests/
|   |-- problem_unittests.py
|
|-- image/
    |-- Face/
        |-- *.jpg
```

---

## Architecture

### Discriminator

The Discriminator is a convolutional neural network that maps a 32x32 RGB image to a single scalar (real/fake probability).

```
Input: (batch, 3, 32, 32)
  Conv2d(3, 64)    + BatchNorm  LeakyReLU   (batch, 64,  16, 16)
  Conv2d(64, 128)  + BatchNorm  LeakyReLU   (batch, 128,  8,  8)
  Conv2d(128, 256) + BatchNorm  LeakyReLU   (batch, 256,  4,  4)
  Conv2d(256, 512)              LeakyReLU   (batch, 512,  2,  2)
  Flatten  Linear(2048, 512)   Dropout(0.3)
  Linear(512, 1)
Output: (batch, 1)
```

### Generator

The Generator maps a latent noise vector z in R^100 to a 32x32 RGB image using transposed convolutions.

```
Input: z (batch, 100)
  Linear(100, 1024)  Reshape (batch, 1024, 1, 1)
  ConvTranspose2d(1024, 512) + BatchNorm  LeakyReLU  (batch, 512,  2,  2)
  ConvTranspose2d(512,  256) + BatchNorm  LeakyReLU  (batch, 256,  4,  4)
  ConvTranspose2d(256,  128) + BatchNorm  LeakyReLU  (batch, 128,  8,  8)
  ConvTranspose2d(128,   64) + BatchNorm  LeakyReLU  (batch,  64, 16, 16)
  ConvTranspose2d(64,     3)              Tanh        (batch,   3, 32, 32)
Output: (batch, 3, 32, 32)
```

---

## Dataset

**CelebA -- Large-scale CelebFaces Attributes Dataset**

- 202,599 face images of celebrities
- 40 binary attribute annotations per image
- Aligned and cropped face versions used here

Download from the [official CelebA page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or via [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

After downloading, place the images inside the `image/Face/` directory. The `ImageFolder` loader from torchvision requires at least one subdirectory inside the root folder.

---

## Model Details

### Weight Initialisation

All `Conv` and `Linear` layers are initialised with weights drawn from a Normal distribution N(0, 0.02) and biases set to 0, following the original DCGAN paper.

### Loss Function

Both networks use **Binary Cross-Entropy with Logits Loss** (`BCEWithLogitsLoss`), which combines a sigmoid layer and BCELoss for numerical stability.

- **Real loss:** Target labels = 1 (real images should be classified as real)
- **Fake loss:** Target labels = 0 (fake images should be classified as fake)

### Input Scaling

Pixel values are scaled from [0, 1] to [-1, 1] to match the Generator's `tanh` output range.

### Optimiser

Both networks use the **Adam optimiser** with a learning rate of 0.0002 and default betas (0.9, 0.999).

---

## Training

### Hyperparameters

| Parameter  | Value  | Description                          |
|------------|--------|--------------------------------------|
| batch_size | 64     | Number of images per training batch  |
| img_size   | 32     | Spatial resolution of images         |
| z_size     | 100    | Dimension of the latent noise vector |
| d_conv_dim | 64     | Base channel count for Discriminator |
| g_conv_dim | 1024   | Base channel count for Generator     |
| n_epochs   | 150    | Total training epochs                |
| lr         | 0.0002 | Learning rate for Adam optimisers    |

### Training Loop

The training loop alternates between updating D and G each batch.

**Step 1 -- Train Discriminator:**
1. Compute real loss on real images. D(real) should output 1.
2. Generate fake images from noise z.
3. Compute fake loss. D(G(z)) should output 0.
4. Total D loss = real loss + fake loss, then backpropagate.

**Step 2 -- Train Generator:**
1. Generate fake images from a new noise vector z.
2. Compute real loss on D(G(z)). G wants D to output 1.
3. Backpropagate through G only.

A fixed `fixed_z` vector is used each epoch to track the visual progress of generated samples, saved to `train_samples.pkl`.

---

## Results

| Metric                | Value        |
|-----------------------|--------------|
| Training Epochs       | 150          |
| Final D Loss (approx) | 0.01 -- 0.30 |
| Final G Loss (approx) | 6.0 -- 9.0   |
| Image Resolution      | 32 x 32 px   |
| Dataset               | CelebA Faces |

---

## References

1. Radford, A., Metz, L., and Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

2. Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

3. Liu, Z. et al. (2015). *Deep Learning Face Attributes in the Wild (CelebA).* ICCV 2015. [Project Page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

4. PyTorch Documentation -- [torch.nn](https://pytorch.org/docs/stable/nn.html)

---

## Licence

This project is released for academic and research purposes. The CelebA dataset is subject to its own [terms of use](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
