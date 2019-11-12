"""モルフォロジー

"""

import cv2
import numpy as np
from scipy.ndimage import measurements


def threshold_otu(image: np.ndarray) -> np.ndarray:
    """大津の2値化

    Args:
      image: 変換元の画像

    Returns:
      変換後の画像

    """
    #: 退色処理
    image_gray, _ = cv2.decolor(image)

    #: 2値化
    image_gray[image_gray < 128] = 16
    image_gray[image_gray >= 128] = 240

    return image_gray


#: 2値化
image: np.ndarray = cv2.imread("imori.jpg")
image = threshold_otu(image)

#: 物体にラベル付け
labels, number_objects = measurements.label(image)
print(f'Number of Objects: {number_objects}')

cv2.imshow("answer03", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
