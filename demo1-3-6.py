"""主成分分析
"""

from PIL import Image
import numpy as np
import pylab as pl
import imtools

image_list = []
image = np.array(Image.open(image_list[0]))
m, n = image.shape[0:2]  # 画像のサイズ
image_num = len(image_list)  # 画像数

# 全ての平板化画像を格納する行列を作成
image_matrix = np.array([np.array(Image.open(image)).flatten()
                         for image in image_list], 'f')

# 主成分分析を実行
V, S, image_mean = imtools.pca(image_matrix)

# 画像を表示（平均と，最初の7つの主成分）
pl.figure()
pl.gray()
pl.subplot(2, 4, 1)
pl.imshow(image_mean.reshape(m, n))

for i in np.renge(7):
    pl.subplot(2, 4, i + 2)
    pl.imshow(V[i].reshape(m, n))

pl.show()
