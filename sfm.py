import numpy as np

def compute_fundamental(x1, x2):
  """
    正規化8点法を使って対応点群(x1, x2: 3*n の配列)から基礎行列を計算
    各列は以下のような並び
      [x'*xm x'*y, y'*x, y'*y, y', x, y, 1]
  """

  n: number = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError("Number of points do not match.")
  
  # 方程式の行列を作成
  A: np.ndarray = np.zeros((n, 9))
  for i in range(n):
    A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
            x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
            x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]
  
  # 線形最小二乗法で計算
  U, S, V = np.linalg.svd(A)
  F = V[-1].reshape(3, 3)

  # Fの制約
  # 最後の特異値を 0 にして階数を 2 にする
  U, S, V = np.linalg.svd(F)
  S[2] = 0
  F = np.dot(U, np.dot(np.diag(S), V))

  return F


def compute_epipole(F):
  """
    基礎行列Fから右側のエピポールを計算
    （左側のエピポールは F.T から計算）
  """

  # F の霊空間(Fx = 0)を返す
  U, S, V = np.linalg.svd(F)
  e = V[-1]
  return e / e[2]