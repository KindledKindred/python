"""Q.01: RGBをBGRに変換

読み込んだイモリの画像について赤と青をひっくり返して出力．

API:
  cv2.BGR2RGB や cv2.RGB2BGR

"""

from PIL import Image
import numpy as np

def X0YtoY0X(image: np.ndarray) -> np.ndarray:
  """1,3番目のチャネルを入れ替え

  読み込んだnumpy.ndarray形式（RGB等）の画像について
  1番目と3番目のチャネルを入れ替える関数です．

  Args:
    image: 変換元の画像

  Returns:
    変換後の画像

  """
  image = image[:, :, ::-1]

  return image

image: np.ndarray = np.array(Image.open('imori.jpg'))
image = X0YtoY0X(image)
image = Image.fromarray(image)
"""
画像をpillowとnp.ndarrayの両形式間で変換しています．

pillow:
  Imageオブジェクト
  image = Image.new(color-mode: string, size: tupple, color: tupple | string)

  Paramaters:
    color-mode: RGBなどのカラーモードを指定
    size: (300, 540)など．原点は左上．
    color: カラーモードに応じた指定か，'red'のようなImageColorModuleに指定された文字列．

OpenCV: 
  np.ndarray（多次元配列）
  
  image = np.zeros((height, width, 3), np.uint8)
  
  np.zerosは零行列を作ります．
  指定したピクセルに unit8 で指定された色を乗せます．
  色は RGB　ではなく BGR の順に指定します．

  詳細は同フォルダ内の opencv_image_nparray.ipynb を参照してください．

"""

image.show()
