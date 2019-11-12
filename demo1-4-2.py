from PIL import Image
import numpy as np
from scipy.ndimage import filters

im = np.array(Image.open('pil_image.png').convert('L'))

#: Sobel微分係数フィルタ
imx = np.zeros(im.shape)  #: imと同サイズの零行列を生成
filters.sobel(im, 1, imx)  #: 元画像に対してx方向(1)にSobelフィルタを適用しimxにコピー

imy = np.zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = np.sqrt(imx**2 + imy**2)

imx = Image.fromarray(imx)
imx.show()

imy = Image.fromarray(imy)
imy.show()

magnitude = Image.fromarray(magnitude)
magnitude.show()
