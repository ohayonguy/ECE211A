from PIL import Image, ImageDraw, ImageFont, ImageChops
from skimage.io._plugins.pil_plugin import pil_to_ndarray
from skimage.io._plugins.pil_plugin import ndarray_to_pil
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import copy

show = False

y_train = Image.open('train.jpg').convert('L').crop((0, 0, 256, 256))
y_test = Image.open('test.jpg').convert('L').crop((0, 0, 256, 256))

if show:
    y_train.save('y_train.png', 'PNG')
    y_test.save('y_test.png', 'PNG')

# Building and saving y_test_missing
y_test_missing = copy.deepcopy(y_test)
draw = ImageDraw.Draw(y_test_missing)
# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype('arial.ttf', size=18)
draw.fontmode = '1'
# draw.text((x, y),"Sample Text",(r,g,b))
draw.text((10, 10),"This Homework is cooly cool",(255), font=font)
draw.text((10, 35),"This Homework is cooly cool",(255), font=font)
draw.text((10, 60),"This Homework is cooly cool",(255), font=font)
draw.text((10, 85),"This Homework is cooly cool",(255), font=font)
draw.text((10, 110),"This Homework is cooly cool",(255), font=font)
draw.text((10, 135),"This Homework is cooly cool",(255), font=font)
draw.text((10, 160),"This Homework is cooly cool",(255), font=font)
draw.text((10, 185),"This Homework is cooly cool",(255), font=font)
draw.text((10, 210),"This Homework is cooly cool",(255), font=font)
draw.text((10, 235),"This Homework is cooly cool",(255), font=font)
draw.text((10, 260),"This Homework is cooly cool",(255), font=font)
y_test_missing.save('y_test_missing.png')

#Building and saving the mask
mask = ImageChops.difference(y_test_missing, y_test)
mask = np.array(pil_to_ndarray(mask) <= 0.1, dtype=int)
mask_img = mask * 255
mask = ndarray_to_pil(mask)
mask_img = ndarray_to_pil(mask_img)
mask_img.save('mask.png', 'PNG')


psnr_y_test_y_test_missing = peak_signal_noise_ratio(pil_to_ndarray(y_test), pil_to_ndarray(y_test_missing))
ssim_y_test_y_test_missing = structural_similarity(pil_to_ndarray(y_test), pil_to_ndarray(y_test_missing))
print(psnr_y_test_y_test_missing)
print(ssim_y_test_y_test_missing)
print('\n')

"""
plt.imshow(y_train, cmap='gray')
plt.show()
plt.imshow(y_test, cmap='gray')
plt.show()
plt.imshow(y_test_missing, cmap='gray')
plt.show()
plt.imshow(mask, cmap='gray')
plt.show()"""


#Create and plot patches
y_test_patches = np.array([pil_to_ndarray(y_test_missing)[i:i + 8, j:j + 8] for j in range(0, 256, 8) for i in range(0, 256, 8)])
mask_patches = np.array([pil_to_ndarray(mask)[i:i+8, j:j+8] for j in range(0, 256, 8) for i in range(0, 256, 8)])

plt.figure(1)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mask_patches[i+60], cmap='gray')
if show:
    plt.show()

plt.figure(2)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_test_patches[i + 60], cmap='gray')
if show:
    plt.show()




#Random dictionary creation and plot
num_of_atoms = 1024
random_patches = [0] * num_of_atoms
for i in range(num_of_atoms):
    random_patches[i] = np.random.randint(0, 255, (8, 8))

num_of_atoms_to_show = 25
plt.figure(3)
for i in range(num_of_atoms_to_show):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(random_patches[i], cmap='gray')
if show:
    plt.show()

#normalize and flatten dictionary
for i in range(num_of_atoms):
    random_patches[i] = random_patches[i] / np.linalg.norm(random_patches[i])
    random_patches[i] = [random_patches[i][k][j] for k in range(8) for j in range(8)]

y_train_patches = np.array([pil_to_ndarray(y_train)[i:i + 8, j:j + 8] for j in range(0, 256, 8) for i in range(0, 256, 8)])


#OMP

#flatten train patches
y_train_patches_flat = [0] * len(y_test_patches)
for i in range(len(y_test_patches)):
    y_train_patches_flat[i] = [y_train_patches_flat[i][k][j] for k in range(8) for j in range(8)]

from sklearn.linear_model import OrthogonalMatchingPursuit
n_nonzero_coefs = 20
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp.fit(random_patches, y_train_patches_flat)
coef = omp.coef_
idx_r = coef.nonzero()
print(idx_r)
#plt.stem(idx_r, coef[idx_r], use_line_collection=True)
