from PIL import Image
import numpy as np
from scipy import ndimage

im = np.array(Image.open('pil_image.png').convert('L'))
im2 = ndimage.filters.gaussian_filter(im, 5)

im2 = Image.fromarray(im2)
im2.show()
