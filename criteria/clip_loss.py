
import torch
import clip
import cv2
import numpy as np
import PIL.Image
from PIL import Image
import io
import torchvision

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
    
    def imshow(self, images, col, viz_size=256):
        """Shows images in one figure."""
        num, height, width, channels = images.shape
        assert num % col == 0
        row = num // col
      
        fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)
      
        for idx, image in enumerate(images):
          i, j = divmod(idx, col)
          y = i * viz_size
          x = j * viz_size
          if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
          fused_image[y:y + viz_size, x:x + viz_size] = image
      
        fused_image = np.asarray(fused_image, dtype=np.uint8)
        data = io.BytesIO()
        PIL.Image.fromarray(fused_image).save(data, 'jpeg')
        im = PIL.Image.fromarray(fused_image)
        return im
    
    def encode(self, image):
        image = self.avg_pool(self.upsample(image))
        return self.model.encode_image(image)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

