# TargetCLIP- official pytorch implementation of the paper Image-Based CLIP-Guided Essence Transfer

This repository finds a *global direction* in StyleGAN's space to edit images according to a target image.
We transfer the essence of a target image to any source image.

## Pretrained directions notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/TargetCLIP/blob/main/TargetCLIP_CLIP_guided_image_essence_transfer.ipynb)

The notebook allows to use the directions on the sources presented in the examples. In addition, there's an option to edit your own inverted images with the pretrained directions, by uploading your latent vector to the `dirs` folder.
We use images inverted by [e4e](https://github.com/omertov/encoder4editing)

## Examples:

*NOTE: all the examples presented are available in our colab notebook. The recommended coefficient to use is between 0.5-1*

### Targets that were not inverted- The Joker and Keanu Reeves 
The targets are plain images, that were not inverted, the direction optimization is initialized at random.

*NOTE: for the joker, we use relatively large coefficients- 0.9-1.3*
<p align="center">
  <img  src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/joker_keanu.jpg">
</p>

### Out of domain targets- Elsa and Pocahontas
The targets are plain images that are out of the domain StyleGAN was trained on, the direction optimization is initialized at random.
<p align="center">
  <img src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/elsa_pocahontas.jpg">
</p>


### Targets that were inverted- Trump
The targets are inverted images, and the latents are used as initialization for the optimization.
<p align="center">
  <img src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/Trump.png">
</p>


### Credits
The code in this repo draws from the [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) code base. 
