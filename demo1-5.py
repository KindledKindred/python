from PIL import Image
import numpy as np
import pylab as pl
import rof

im = np.array(Image.open("pil_image.png").convert('L'))
U, T = rof.denoise(im, im)

pl.figure()
pl.gray()
pl.imshow(U)
pl.axis('equal')
pl.axis('off')
pl.show()
