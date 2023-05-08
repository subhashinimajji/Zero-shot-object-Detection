# -*- coding: utf-8 -*-
"""Copy of MINI .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wuo5jz-3R06LOuNBGsAeQwMpcaG_gFlR
"""

pip install datasets

# import dataset
from datasets import load_dataset

data = load_dataset(
    "jamescalam/image-text-demo",
    split="train",
    revision="180fdae",
)
data

type(data[2]['image'])

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load the image from a URL or file path
img = data[2]['image']

# Plot the image using Matplotlib
plt.imshow(img)
plt.axis('off')
plt.show()

# Download the image
img.save('image.jpg')
from google.colab import files
files.download('image.jpg')

from torchvision import transforms

# transform the image into tensor
transt = transforms.ToTensor()
data = Image.open("image.jpg")

img = transt(data)
img.data.shape

# add batch dimension and shift color channels
patches = img.data.unfold(0,3,3)
patches.shape

# break the image into patches (in height dimension)
patch = 256

new_patches = patches.unfold(1, patch, patch)
new_patches.shape

# break the image into patches (in width dimension)
new_patches_1 = new_patches.unfold(2, patch, patch)
new_patches_1.shape

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

window = 6
stride = 1
# print(0, new_patches_1.shape[1]-window+1, stride)

# window slides from top to bottom
for Y in range(0, new_patches_1.shape[1]-window+1, stride):
    # window slides from left to right
    
    for X in range(0, new_patches_1.shape[2]-window+1, stride):
        # initialize an empty big_patch array
        
        big_patch = torch.zeros(patch*window, patch*window, 3)
        # this gets the current batch of patches that will make big_batch
        patch_batch = new_patches_1[0, Y:Y+window, X:X+window]
        # loop through each patch in current batch
        for y in range(patch_batch.shape[1]):
            for x in range(patch_batch.shape[0]):
                # add patch to big_patch
                big_patch[
                    y*patch:(y+1)*patch, x*patch:(x+1)*patch, :
                ] = patch_batch[y, x].permute(1, 2, 0)
        # display current big_patch
#         print(patch_batch.shape)
        # plt.imshow(big_patch)
        # plt.show()

pip install transformers

from transformers import CLIPProcessor, CLIPModel
import torch

# define processor and model
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

window = 6
stride = 1

scores = torch.zeros(new_patches_1.shape[1], new_patches_1.shape[2])
runs = torch.ones(new_patches_1.shape[1], new_patches_1.shape[2])

for Y in range(0, new_patches_1.shape[1]-window+1, stride):
    for X in range(0, new_patches_1.shape[2]-window+1, stride):
        big_patch = torch.zeros(patch*window, patch*window, 3)
        patch_batch = new_patches_1[0, Y:Y+window, X:X+window]
        for y in range(window):
            for x in range(window):
                big_patch[
                    y*patch:(y+1)*patch, x*patch:(x+1)*patch, :
                ] = patch_batch[y, x].permute(1, 2, 0)
        # we preprocess the image and class label with the CLIP processor
        inputs = processor(
            images=big_patch,  # big patch image sent to CLIP
            return_tensors="pt",  # tell CLIP to return pytorch tensor
            text="an animal",  # class label sent to CLIP
            padding=True
        ).to(device) # move to device if possible

        # calculate and retrieve similarity score
        score = model(**inputs).logits_per_image.item()
        # sum up similarity scores from current and previous big patches
        # that were calculated for patches within the current window
        scores[Y:Y+window, X:X+window] += score
        # calculate the number of runs on each patch within the current window
        runs[Y:Y+window, X:X+window] += 1

scores /= runs

import numpy as np

# clip the scores
scores = np.clip(scores-scores.mean(), 0, np.inf)

# normalize scores
scores = (
    scores - scores.min()) / (scores.max() - scores.min()
)

scores.shape, new_patches_1.shape

# transform the patches tensor
adj_patches = new_patches_1.squeeze(0).permute(3, 4, 2, 0, 1)
adj_patches.shape

# multiply patches by scores
adj_patches = adj_patches * scores

# rotate patches to visualize
adj_patches = adj_patches.permute(3, 4, 2, 0, 1)
adj_patches.shape

Y = adj_patches.shape[0]
X = adj_patches.shape[1]

fig, ax = plt.subplots(Y, X, figsize=(X*.5, Y*.5))
for y in range(Y):
    for x in range(X):
        ax[y, x].imshow(adj_patches[y, x].permute(1, 2, 0))
        ax[y, x].axis("off")
        ax[y, x].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# scores higher than 0.5 are positive
detection = scores > 0.5

# non-zero positions
np.nonzero(detection)

y_min, y_max = (
    np.nonzero(detection)[:,0].min().item(),
    np.nonzero(detection)[:,0].max().item()+1
)
y_min, y_max

x_min, x_max = (
    np.nonzero(detection)[:,1].min().item(),
    np.nonzero(detection)[:,1].max().item()+1
)
x_min, x_max

y_min *= patch
y_max *= patch
x_min *= patch
x_max *= patch
x_min, y_min

height = y_max - y_min
width = x_max - x_min

height, width

# image shape
img.data.numpy().shape

# move color channel to final dim
image = np.moveaxis(img.data.numpy(), 0, -1)
image.shape

import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))

ax.imshow(image)

# Create a Rectangle patch
rect = patches.Rectangle(
    (x_min, y_min), width, height,
    linewidth=3, edgecolor='#FAFF00', facecolor='none'
)

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()





