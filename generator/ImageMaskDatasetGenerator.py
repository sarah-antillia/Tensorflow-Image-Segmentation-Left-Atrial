# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/04/02 ImageMaskBlender.py

import os
import io
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback
import matplotlib.pyplot as plt

import cv2

#RESIZE = 512

class ImageMaskDatasetGenerator:

  def __init__(self, 
               images_dir = "",
               masks_dir  = "", 
               output_dir = "", 
               angle      = 90,
               mirror     = True,
               resize     = 512):
    
    self.images_dir = images_dir 
    self.masks_dir   = masks_dir
    self.mirror     = mirror
    self.angle      = angle
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    if not os.path.exists(self.output_images_dir):
      os.makedirs(self.output_images_dir)

    if not os.path.exists(self.output_masks_dir):
      os.makedirs(self.output_masks_dir)
    self.BASE_INDEX = 1000
    self.RESIZE    = (resize, resize)

  def generate(self):
      image_files = glob.glob(self.images_dir + "/*.nii")
      mask_files  = glob.glob(self.masks_dir  + "/*.nii")
      image_files = sorted(image_files)
      mask_files  = sorted(mask_files)

      for mask_file in mask_files:
        self.generate_mask_files(mask_file )
      for image_file in image_files: 
        self.generate_image_files(image_file)
    
  def fig2image(self, fig):
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return Image.frombytes( "RGB", ( w ,h ), buf.tostring( ) )
  
 
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  # Modified to save plt-image to BytesIO() not to a file.
  def generate_image_files(self, nii_file):
    nii = nib.load(nii_file)
    basename = os.path.basename(nii_file) 
    nameonly = basename.split(".")[0] 
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = nameonly + "_" + str(i+ self.BASE_INDEX) + ".jpg"
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
        img   = self.normalize(img)
        image = Image.fromarray(img)
        image = image.convert("RGB")
 
        image = image.resize(self.RESIZE)
        if self.angle>0:
          image = image.rotate(self.angle)
        if self.mirror:
          image = ImageOps.mirror(image)

        image.save(filepath)
        print("=== Saved {}".format(filepath))
     

  def generate_mask_files(self, nii_file ):
    nii = nib.load(nii_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      basename = os.path.basename(nii_file) 
      nameonly = basename.split(".")[0]
      if img.any() >0:
        img = img*255.0
        img = img.astype('uint8')

        image = Image.fromarray(img)
        image = image.convert("RGB")
        image = image.resize(self.RESIZE)
 
        if self.angle >0:
          image = image.rotate(self.angle)
        if self.mirror:
          image = ImageOps.mirror(image)
 
        filename  = nameonly + "_" + str(i+ self.BASE_INDEX) + ".jpg"
        filepath  = os.path.join(self.output_masks_dir, filename)
        image.save(filepath, "JPEG")
        print("--- Saved {}".format(filepath))


if __name__ == "__main__":
  try:
    images_dir  = "./imagesTr/"
    masks_dir   = "./labelsTr/"
    output_dir = "./Left-Atrial-master"
    mirror     = True
    angle      = 90

    generator = ImageMaskDatasetGenerator(images_dir=images_dir, 
                                          masks_dir =masks_dir, 
                                          output_dir=output_dir,
                                          mirror = mirror,
                                          angle = angle)
    generator.generate()
  except:
    traceback.print_exc()

 
