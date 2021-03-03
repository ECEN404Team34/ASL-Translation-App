# Sample program provided by HandSeg to display data - interactive so does not run on olympus

from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
matplotlib.use('pdf')

depth_im = np.array(Image.open('/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/data/images/user-4.00000003.png'))  # Loading depth image
mask_im = np.array(Image.open('/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/data/masks/user-4.00000003.png'))  # Loading mask image
depth_im = depth_im.astype(np.float32)  # Converting to float
mean_depth_ims = 10000.0   # Mean value of the depth images
depth_im /= mean_depth_ims  # Normalizing depth image
plt.imshow(depth_im); plt.title('Depth Image'); plt.savefig('depth.pdf')  # Displaying Depth Image
plt.imshow(mask_im); plt.title('Mask Image'); plt.show('mask.pdf')  # Displaying Mask Image
