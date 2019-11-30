# サンプルデータ: http //www. robots.ox.ac.uk/ ̃vgg/data/data-mview.html

import numpy as np
import pylab as pl
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
!pip install pygame
import pygame, pygame.image
from pygame.locals import *
import matplotlib as plt
from mpl_toolkits.mplot3d import axes3d

class Camera (object):
  """ ピンホールカメラを表すクラス """
  def __init__(self, P):
    """ カメラモデル P = K[R|t] を初期化 """
    self.P = P
    self.K = None # キャリブレーション行列
    self.R = None # Rotation
    self.t = None # 平行移動
    self.c = None # カメラ中心
  
  def project(self, X):
    """ X(4*n の配列)の点を射影し，座標を正規化 """
    x = np.dot(self.P, X)
    for i in range(3):
      x[i] /= x[2]
    return x
  
  def rotation_matrix(a):
    """ ベクトルaを軸に回転する3Dの回転行列をreturn """
    R = np.eye(4)
    R[:3, :3] = np.linalg.expm(
        [[0, -a[2], a[1]],
         [a[2], 0, -a[0]],
         [-a[1], a[0], 0]]
        )
    return R
  
  def factor(self):
    """ P = K[R|t] に従い，カメラ行列を K, R, t に分解 """
    # 最初の 3*3 の部分を分解
    K, R = np.linalg.rq(self.P[:, :3])

    # Kの対角成分が正になるようにする
    T = np.diag(sign(diag(K)))
    if np.linalg.det(T) < 0:
      T[1, 1] *= -1
    
    self.K = np.dot(K, T)
    self.R = np.dot(T, R) # Tはそれ自身が逆行列
    self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

    return self.K, self.R, self.t
  
  def center(self):
    """ カメラ中心を求める """
    if self.c is not None:
      return self.c
    else:
      # 分解により計算
      self.factor()
      self.c = -np.dot(self.R.T, self.t)
      return self.c
    
  def my_calibration(sz):
    """ カメラ定数を求める """
    row, col = sz
    fx = 2555.0 * col / 2592
    fy = 2586.0 * row / 1936
    K = np.diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    return K
  
  def cube_points(c, wid):
    """ 
      plotで立方体を描画するための頂点のリストを生成
      最初の5点は底面の正方形であり，辺が繰り返される
    """
    p = []

    # 底面
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid]) # 描画を閉じる

    # 上面
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid]) # 描画を閉じる

    # 垂直の辺
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return np.array(p).T

  def set_projection_from_camera(K):
    """ カメラのキャリブレーション行列から表示領域を設定 """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * np.arctan(0.5 * height / fy) * 180 / pi
    aspect = floar(width * fy) / (height * fx)

    # 手前と奥のクリッピング平面を定義
    near = 0.1
    far = 100.0

    # 射影を設定
    gluPerspective(fovy, aspect, near, far)
    flViewport(0, 0, width, height)
  
  def set_modelview_from_camera(Rt):
    """ カメラの姿勢からモデルビュー行列を設定 """
    glMatrixMode(GL_MODEVIEW)
    glLoadIdentity()

    # ティーポットをx軸周りに90度回転させz軸を上向きに
    Rx = np.array([1, 0, 0], [0, 0, -1], [0, 1, 0])

    # 回転を最適な近似に
    R = Rt[:, :3]
    U, S, V = np.linalg.svd(R)
    R = np.dot(U, V)
    R[0, :] = - R[0, :] # x軸の符号反転

    # 平行移動
    t = Rt[:, 3]

    # 4*4 のモデルビュー行列
    M = np.eye(4)
    M[:3, :3] = np.dot(R, Rx)
    M[:3, 3] = t

    # 列方向に平板化するために転置
    M = M.T
    m = M.flatten()

    # モデルビュー行列を新しい行列に置き換え
    glLoadMatrixf(m)


"""
  特徴点の描画
"""

image1: np.ndarray = np.array(Image.open('images/001.jpg'))
image2: np.ndarray = np.array(Image.open('images/002.jpg'))

points2D = [np.loadtxt('2D/00' + str(i + 1) + '.corners').T for i in range(3)]
points3D = np.loadtxt('3D/p3d').T

corr = np.genfromtxt('2D/nview-corners', dtype='int', missing_values='*')

print(np.loadtxt('2D/001.P'))
P = [Camera(np.loadtxt('2D/00'+ str(i + 1) + '.P')) for i in range(3)]

# 3Dの点を同次座標にして描画
X = np.vstack((points3D, np.ones(points3D.shape[1])))
x = P[0].project(X)

# 画像1の上に点を描画
pl.figure()
pl.imshow(image1)
pl.plot(points2D[0][0], points2D[0][1], '*')
pl.axis('off')

pl.figure()
pl.imshow(image1)
pl.plot(x[0], x[1], 'r.')
pl.axis('off')

pl.show()


"""
  3Dデータの描画
"""

axes = pl.figure().gca(projection="3d")

# 3Dのサンプルデータを生成
X, Y, Z = axes3d.get_test_data(0.25)

# 3Dの点を描画
axes.plot(X.flatten(), Y.flatten(), Z.flatten(), 'o')

pl.show()