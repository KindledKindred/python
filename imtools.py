from PIL import Image
import numpy as np
import pylab as pl


def imageResize(image: np.ndarray, size) -> np.ndarray:
    """リサイズ

    Args:
      image: リサイズしたい画像
      size: リサイズされるサイズ

    Returns
      リサイズ後の画像

    """
    pil_image = Image.fromarray(np.uint8(image))
    return np.array(pil_image.resize(size))


def histequal(image: np.ndarray, nbr_bins: int = 256):
    """ヒストグラム平坦化

    すべての明度が同程度になるような変換法．
    ヒストグラムの累積分布を正規化することで実現．

    Args:
      image: 変換元の画像
      nbr_bins: ヒストグラムのx軸（RGBの場合256）

    Returns:
      ヒストグラム平坦化後の画像, 正規化された累積分布関数

    """
    imagehist, bins = pl.histogram(image.flatten(), nbr_bins, normed=True)
    cdf = imagehist.cumsum()  # 累積分散関数
    cdf = 255 * cdf / cdf[-1]  # 正規化

    image_after = np.interp(image.flatten(), bins[:-1], cdf)  # 線形補完
    return image_after.reshape(image.shape), cdf


def compute_average(image_list):
    """平均画像

    Args:
      image_list: 読み込みたい画像のファイル名が入った配列

    Returns:
      読み込んだ全画像の平均
    """
    average_image = np.array(Image.open(image_list[0]), 'f')

    for image_name in image_list[1:]:
        try:
            average_image += np.array(Image.open(image_name))
        except FileNotFoundError:
            print(f'{image_name} is skipped')

    average_image /= len(image_list)

    return np.array(average_image, 'uint8')


def pca(X):
    """主成分分析

    Args:
      X: 入力データを平板化した配列を行として格納した行列

    Returns:
      写像行列（次元の重要度順）, 分散, 平均

    """

    # 次元数を取得
    num_data, dim = X.shape

    # データをセンタリング
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - 高次元の時はコンパクトな裏技を用いる
        M = np.dot(X, X.T)  # 共分散行列
        e, EV = np.linalg.eigh(M)  # 固有値と固有ベクトル
        tmp = np.dot(X.T, EV).T  # ここがコンパクトな裏技
        V = tmp[::-1]  # 末尾の固有ベクトルほど重要なので，反転する
        S = np.sqrt(e)[::-1]  # 固有値の並びも反転する

        for i in range(V.shape[1]):
            V[:, i] /= S

    else:
        # PCA - 低次元なら特異値分解を用いる
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # 最初のnum_dataだけが有用

    return V, S, mean_X
