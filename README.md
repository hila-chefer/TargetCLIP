# TargetCLIP- official pytorch implementation of the paper [Image-Based CLIP-Guided Essence Transfer](https://arxiv.org/abs/2110.12427)

This repository finds a *global direction* in StyleGAN's space to edit images according to a target image.
We transfer the essence of a target image to any source image.

## Pretrained directions notebooks:
### Notebook for celebrity sources/ your own pre-inverted latents:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/TargetCLIP/blob/main/TargetCLIP_CLIP_guided_image_essence_transfer.ipynb)

The notebook allows to use the directions on the sources presented in the examples. In addition, there's an option to edit your own inverted images with the pretrained directions, by uploading your latent vector to the `dirs` folder.
We use images inverted by [e4e](https://github.com/omertov/encoder4editing).

### Notebook for e4e+TargetCLIP (inversion and manipulation in one notebook):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/TargetCLIP/blob/main/TargetCLIP%2Be4e.ipynb)


## Training new directions:
To train new directions for your own targets, use the `find_dirs.py` script under the `optimization` folder.

Our code supports both targets from images the were **not inverted** and targets for inverted images. For example, our Elsa, The Joker, Pocahontas, Keanu Reeves, and more examples were not inverted, while our Trump example was inverted.
When possible, an inverted target usually gives better results.

We recommend to use inverted images for the training process. Our experiments use [the inverted latents from the StyleCLIP repo](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view).

### Using targets that were not inverted
The code uses `--num_directions` differnet random initializations for the essence vector. After training, you can choose your favorite one (usually, all are very similar).
1. Download [the inverted latents from the StyleCLIP repo](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view) for training.
2. Upload your target image to the `dirs/tragets` folder. Note that png images are not supported.
3. Run the `find_dirs.py` script with your target:
```
PYTHONPATH=`pwd` python optimization/find_dirs.py --target_path dirs/targets/your_target.jpg --dir_name results_folder --weight_decay 3e-3 --lambda_consistency 0.6 --step 1000 --lr 0.2 --num_directions 8 --num_images 8 --data_path path_to_styleclip_latents
```

The intermediate best results for your training samples will appear under the path specified in `--dir_name`. In addition, the optimal essence vectors for all your initializations will be saved as `direction{i}.npy`, and you can use them on other images or upload them to the notebook to experiment with other sources.

**Note:** for targets that require extreme or difficult semantic edits (e.g. avatar, thanos, etc.), try to increase the influence of the transfer loss using the `lambda_transfer` argument (default is set to 1).
### Using inverted targets
We will initialize the essence vector to be the latent of your target.
1. Download [the inverted latents from the StyleCLIP repo](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view) for training.
2. Upload your target's latent to the `dirs/tragets` folder. We use [e4e](https://github.com/omertov/encoder4editing) to invert all our images.
3. Run the `find_dirs.py` script with your target latnet:

```
PYTHONPATH=`pwd` python optimization/find_dirs.py  --dir_initialition dirs/tragets/your_target.pt --num_directions 8  --num_images 8 --dir_name results_folder --weight_decay 3e-3 --lambda_consistency 0.6 --step 1000 --lr 0.2 --data_path path_to_styleclip_latents
```

The intermediate best results for your training samples will appear under the path specified in `--dir_name`. In addition, the optimal essence vectors for all your initializations will be saved as `direction0.npy`, which is the essence vector derived from your input latent.


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

## Updates:

10/27/21: Pretrained directions added for Doc Brown (Back to the Future), Morgan Freeman, Beyonce, and Ariel (The Little Mermaid)!
<p align="center">
  <img  src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/doc_brown_morgan.jpg">
</p>
<p align="center">
  <img  src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/ariel_beyonce.jpg">
</p>

11/2/21: Pretrained directions added for Wolverine, Avatar, and Gargamel!
<p align="center">
  <img  src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/Wolverine_avatar.jpg">
</p>

11/12/21: New pretrained directions added for Ed Sheeran, Dumbledore, Moana, Zendaya, Thanos, and more!
<p align="center">
  <img height=700  src="https://github.com/hila-chefer/TargetCLIP/blob/main/examples/new_directions.jpg">
</p>




## Citing our paper
If you make use of our work, please cite our paper:
```
@article{chefer2021targetclip,
  title={Image-Based CLIP-Guided Essence Transfer},
  author={Chefer, Hila and Benaim, Sagie and Paiss, Roni and Wolf, Lior},
  journal={arXiv preprint arXiv: 2110.12427},
  year={2021}
}
```

### Credits
The code in this repo draws from the [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) code base. 
