import argparse
import os

import cv2
import numpy as np
import torch

import glob
import random

from experiments.Losses import IDLoss
from utils.common import tensor2im
from models.psp import pSp
from argparse import Namespace
from models.stylegan2.model import Generator
import torchvision
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from PIL import Image, ImageOps
import requests
import torch
from torchvision import transforms
from BLIP.models.blip import blip_feature_extractor
import dlib
from utils.alignment import align_face
from utils.common import tensor2im
from criteria.clip_loss import CLIPLoss
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

clip_loss = CLIPLoss(1024)

def cv2ten(img, device):
    img = (img[:, :, ::-1].transpose(2, 0, 1) / 255. - 0.5) / 0.5
    img_ten = torch.from_numpy(img).float().unsqueeze(0).to(device)
    return img_ten


def ten2cv(img_ten, bgr=True):
    img = img_ten.squeeze(0).mul_(0.5).add_(0.5).mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if bgr:
        img = img[:, :, ::-1]
    return img

def get_blip(args, device):
    args.blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    blip = blip_feature_extractor(pretrained=args.blip_url, image_size=args.image_size, vit='base')
    blip.eval()
    blip = blip.to(device)
    return blip

resize_dims = (256, 256)

def load_image(image_size, device, raw_image):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def load_model(args):
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    return g_ema

def get_essence(net, image_path, EXPERIMENT_ARGS):
    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    input_image = run_alignment(image_path)
    input_image.resize(resize_dims)
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image, latent = images[0], latents[0]
    return latents

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_clip_transform():
    return Compose([
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])


def get_semantic_metrics(args, device, source_amp, target, blip, source_amp_pil=None):
    transform = get_clip_transform()
    if source_amp_pil is None:
        source_amp_pil = tensor2im(source_amp.reshape(3, 1024, 1024))
    target_pil = tensor2im(target.reshape(3, target.shape[-2], target.shape[-1]))

    # calculate semantic metrics
    # BLIP
    source_amp_blip = load_image(image_size=args.image_size, device=device, raw_image=source_amp_pil)
    target_blip = load_image(image_size=args.image_size, device=device, raw_image=target_pil)
    target_features_blip = blip(target_blip, '', mode='image')[0, 0]
    src_amp_features_blip = blip(source_amp_blip, '', mode='image')[0, 0]

    # CLIP
    source_amp_clip = transform(source_amp_pil).unsqueeze(0)
    source_amp_clip = clip_loss.model.encode_image(source_amp_clip).cuda()
    target_transform = transform(target_pil).cuda().unsqueeze(0)
    target_clip = clip_loss.model.encode_image(target_transform)
    src_amp_features_clip = source_amp_clip.flatten()
    target_features_clip = target_clip.flatten()

    return src_amp_features_blip, target_features_blip, src_amp_features_clip, target_features_clip