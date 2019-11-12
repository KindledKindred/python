from scipy import ndimage
from PIL import Image
import numpy as np
import pylab as pl


def image_in_image(image_material, image_canvas):
    """
        四隅をできるだけ同次座標tpに近づけるアフィン変換を使って
        image1をimage2に埋め込む．
    """
    #: アフィン変換を適用
    image_alpha = (ndimage.affine_transform(
        image_material, [[1, 1], [1, 1]], (30, 200)) > 0)

    return (1 - image_alpha) * image_canvas + image_alpha * image_material


image_material = np.array(Image.open('IMG_0490.png').convert('L'))
image_canvas = np.array(Image.open('IMG_0491.png').convert('L'))

images = image_in_image(image_material, image_canvas)

pl.figure()
pl.gray()
pl.imshow(images)
pl.axis('equal')
pl.axis('off')
pl.show()
