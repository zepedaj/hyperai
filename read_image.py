from spectral.io.bsqfile import BsqFile
from spectral.io.envi import *
import numpy as np
import cv2
from spectral import *

# read image hyper spectral
header_path='./data/SSM5x5-NIR_09580554_20160605171039.hdr'
filename='./data/SSM5x5-NIR_09580554_20160605171039.bsq'

h = read_envi_header(header_path)
p = gen_params(h)
p.filename=filename
img_hyper = BsqFile(p, h)

height, width, channels = img_hyper.shape
img_hyper = img_hyper[:,:,:]
print(img_hyper.shape)

# read hyperspectral mask
path_hyper_mask="./data/hyper_apples_labels.npy"
mask_hyper=np.load(path_hyper_mask)
print(mask_hyper.shape)

#read png image
path_rgb_image="./data/rgb_apples.png"
img_rgb=cv2.imread(path_rgb_image)
print(img_rgb.shape)
