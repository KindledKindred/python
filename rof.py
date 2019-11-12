import numpy as np


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """
    ROFノイズ除去モデルの実装

    Args:
      im: ノイズのあるグレースケール画像
      U_init: Uの初期ガウス分布
      tolerance: 終了判定基準の許容誤差
      tau: ステップ長
      tv_weight: TV正規化項の重み

    Returns:
      U: ノイズ除去された画像
      im-U: 残余テクスチャ
    """

    m, n = im.shape

    #: init
    U = U_init
    Px = im  #: 双対領域でのx成分
    Py = im  #: 双対領域でのy成分
    """
    双対領域
    　例えば色の塗分けにおいて隣り合う色が別の色になるときの、そのお互いの領域のこと
    """
    error = 1

    while (error > tolerance):
        U_old = U

        #: 主変数の勾配
        GrandUx = np.roll(U, -1, axis=1) - U  #: Uの勾配のx成分
        GrandUy = np.roll(U, -1, axis=0) - U  #: Uの勾配のy成分

        #: 双対変数の更新
        """
        双対問題:
        　主問題の対となる問題のこと

        双対変数:
        　双対問題、特にラグランジュ双対問題の変数のこと
        """
        Px_new = Px + (tau / tv_weight) * GrandUx
        Py_new = Py + (tau / tv_weight) * GrandUy
        Norm_new = np.maximum(1, np.sqrt(Px_new ** 2 + Py_new ** 2))

        Px = Px_new / Norm_new
        Py = Py_new / Norm_new

        #: 主変数の更新
        RxPx = np.roll(Px, 1, axis=1)  #: x成分の右回り変換
        RyPy = np.roll(Py, 1, axis=0)  #: y成分の右回り変換

        DivP = (Px - RxPx) + (Py - RyPy)  #: 双対領域の発散

        U = im + tv_weight * DivP

        #: 誤差の更新
        error = np.linalg.norm(U - U_old) / np.sqrt(n * m)

    return U, im-U
