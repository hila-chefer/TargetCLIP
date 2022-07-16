# [ECCV 2022] TargetCLIP- official pytorch implementation of the paper [Image-Based CLIP-Guided Essence Transfer](https://arxiv.org/abs/2110.12427)

This repository finds a *global direction* in StyleGAN's space to edit images according to a target image.
We transfer the essence of a target image to any source image.

## Pretrained directions notebooks:
### Notebook for celebrity sources/ your own pre-inverted latents:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/TargetCLIP/blob/main/TargetCLIP_CLIP_guided_image_essence_transfer.ipynb)

The notebook allows to use the directions on the sources presented in the examples. In addition, there's an option to edit your own inverted images with the pretrained directions, by uploading your latent vector to the `dirs` folder.
We use images inverted by [e4e](https://github.com/omertov/encoder4editing).

### Notebook for e4e+TargetCLIP (inversion and manipulation in one notebook):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hila-chefer/TargetCLIP/blob/main/TargetCLIP%2Be4e.ipynb)

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

## Downloading pretrained weights 
First, please download all the pretrained weights for the experiments to the folder `pretrained_models`. If you choose to save the pretrained weights in another path, please update the config file accordingly (`configs/paths_config.py`). 
Ours tests require downloading the [pretrained StyleGAN2 weights](https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT), and the [pretrained ArcFace weights](https://github.com/TreB1eN/InsightFace_Pytorch). For our encoder finetuning, please download the [e4e pretrained weights](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view).

To enable alignment, run the following:
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```

## Training the optimizer and the encoder
### Downloading datasets
The targets for our celebrities test [can be found here](https://drive.google.com/drive/folders/1MdY-_lcs5l1v1MwG2mwg-p_nv_sR2oF3). 
To train the encoder, please download the CelebA-HQ dataset (both the test set and the train set), and for the FFHQ tests, download the FFHQ train set as well, and extract the first 50 images from it. 

### Training the encoder from scratch
1. Download ninja=1.10.0, using the following commands:
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```
2. Randomly select 200 images from the CelebsHQ train set and place them in: `data/celeba_minimized`.
3. Randomly select 50 images from the CelebsHQ test set and place them in: `data/data1024x1024/test`.
4. We train our encoder on 5 RTX 2080 Ti GPUs with 11 GB per each GPU. To train the encoder from scratch, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 PYTHONPATH=`pwd` python scripts/train.py --exp_dir name/of/experiment/directory --lambda_consistency 0.5 --batch_size 1 --test_batch_size 1 --lambda_reg 3e-3 --checkpoint_path pretrained_models/e4e_ffhq_encode.pt --image_interval 1 --board_interval 5 --val_interval 31 --dataset_type celeba_encode_minimized --save_interval 200 --max_steps 3000

```
If you wish to train the encoder with a single GPU, please remove the use of `DataParallel` in the coach file (`training/coach`).
The best checkpoint will be saved to `name/of/experiment/directory/checkpoints`.

**Important: Please make sure to download the pretrained [e4e weights](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view) before training in order to enable the finetuning.**

### Training directions with the optimizer
Run the following command:

```
PYTHONPATH=`pwd` python optimization.py --target_path /path/to/target/image --output_folder path/to/optimizer/output  --lambda_transfer 1 --weight_decay 3e-3 --lambda_consistency 0.5 --step 1000 --lr 0.2 --num_directions 1 --num_images 4 
```
where `num_directions` is the number of different directions you wish to train, and `num_images` is the number of images to use in the consistency tests.
Use the `random_initiate` parameter to initialize the direction randomly instead of the inversion of the target.
The result manipulations on the training sources, as well as the produced essence directions will be saved under `output_folder`.

**Important: Please use the targets before alignment (i.e. use the targets from `target_celebs`), as the code for the optimization performs alignment as its first step.**

## Producing quantitative results (id scores, semantic scores)
### Encoder
1. The latents for our 68 sources are saved under pretrained_weights/celebs.pt.
2. Use the method to produce a manipulation for each source, target, and save the results under a folder with the baseline name. The naming convention our tests expect is: `{target_name}/{source_idx}.png` for exmaple, the manipulation for ariel with source number 1 will be saved as: `ariel/1.png`.
3. Produce results by running the following command:
```
PYTHONPATH=`pwd` python ./experiments/calc_metrics.py --style_img_path /path/to/target/images --manipulations_path /output/folder --input_img_path /path/to/source/images
```
where `style_img_path` is the path to the target images, `manipulations_path` is the path to the results of the manipulations, and `input_img_path` is the path to the 68 source images.

## Producing FID
After producing the quantitative scores in the steps above, all the results will be saved to a folder with the name passed in the parameter `outdir`.

For each target, `outdir` will contain a folder with the 68 results of the manipulations with the method, by the target.
For example, in the celebrities test, if our outdir is out, the previous step creates a folder `out/ariel`.

To run the FID test, follow these steps:
1. [Install the FID calculation package](https://github.com/mseitzer/pytorch-fid).
2. Extract a random subset of size 7000 from the FFHQ test set.
3. For each target name, the folder `outdir/target_name` needs to be compared to the subset of FFHQ:
```
python -m pytorch_fid --device cuda:{gpu_device} /path/to/FFHQ /outdir/target_name
```
4. Calculate the avergae and standard deviation across the FID scores of all targets.




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
