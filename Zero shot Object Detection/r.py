import os
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# define processor and model
# Load the saved model and tokenizer from the directory

# define processor and model
model_id = "model" 

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)


# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# transform the image into tensor
transt = transforms.ToTensor()
data = Image.open("image.jpg")

img = transt(data)
# add batch dimension and shift color channels
patches = img.data.unfold(0,3,3)
# break the image into patches (in height dimension)
patch = 256
new_patches = patches.unfold(1, patch, patch)
# break the image into patches (in width dimension)
new_patches_1 = new_patches.unfold(2, patch, patch)
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
# clip the scores
scores = np.clip(scores-scores.mean(), 0, np.inf)

# normalize scores
scores = (
    scores - scores.min()) / (scores.max() - scores.min()
)
# transform the patches tensor
adj_patches = new_patches_1.squeeze(0).permute(3, 4, 2, 0, 1)

# multiply patches by scores
adj_patches = adj_patches * scores

# rotate patches to visualize
adj_patches = adj_patches.permute(3, 4, 2, 0, 1)

# rotate patches to visualize
adj_patches = adj_patches.permute(3, 4, 2, 0, 1)
Y = adj_patches.shape[0]
X = adj_patches.shape[1]

# fig, ax = plt.subplots(Y, X, figsize=(X*.5, Y*.5))
# for y in range(Y):
#     for x in range(X):
#         ax[y, x].imshow(adj_patches[y, x].permute(1, 2, 0))
#         ax[y, x].axis("off")
#         ax[y, x].set_aspect('equal')
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# scores higher than 0.5 are positive
detection = scores > 0.5
# non-zero positions
np.nonzero(detection)

y_min, y_max = (
    np.nonzero(detection)[:,0].min().item(),
    np.nonzero(detection)[:,0].max().item()+1
)

x_min, x_max = (
    np.nonzero(detection)[:,1].min().item(),
    np.nonzero(detection)[:,1].max().item()+1
)

y_min *= patch
y_max *= patch
x_min *= patch
x_max *= patch

height = y_max - y_min
width = x_max - x_min

# move color channel to final dim
image = np.moveaxis(img.data.numpy(), 0, -1)

print(image.shape)

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

# Save the plot to a bytes buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
    
# Clear the plot and close the figure
plt.clf()
plt.close()