from scipy import ndimage
from PIL import Image
import numpy as np
import pylab as pl

image = np.array(Image.open('pil_image.png').convert('L'))
H = np.array([[1.2, 0.03, -100], [0.05, 1.5, -100], [0, 0, 1]])
image2 = ndimage.affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))

pl.figure()
pl.gray()
pl.imshow(image2)
pl.show()
