import numpy as np
import matplotlib as plt
from scipy.ndimage import filters


def compute_harris_response(im, sigma=3):
    """
        グレースケール画像の各ピクセルについて
        Harrisコーナー検出器の応答関数を定義
    """

    #: 微分係数
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    #: Harris行列の成分
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    #: 判別色と対角成分
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtrace = Wxx + Wyy

    return Wdet / Wtrace


def get_harris_points(harris_image, min_dist=10, threshold=0.1):
    """
        Harris応答画像からコーナーを返す

        Args:
            harris_image: Harris応答画像
            min_dist: コーナーや画像境界から分離する最小ピクセル数
    """
    #: 閾値を超えるコーナー候補を見つける
    corner_threshold = harris_image.max() * threshold
    harris_image_t = (harris_image > corner_threshold) * 1

    #: 候補の座標を得る
    coords = np.array(harris_image_t.nonzero()).T

    #: 候補の値を得る
    candidate_values = [harris_image[c[0], c[1]] for c in coords]

    #: 候補をソートする
    index = np.argsort(candidate_values)

    #: 許容する点の座標を配列に格納する
    allowed_locations = np.zeros(harris_image.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    #: 最小距離を考慮しながら最良の点を得る
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
                (coords[i, 0] - min_dist): coords[i, 0] + min_dist,
                (coords[i, 1] - min_dist): (coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """
        画像中に見つかったコーナーを描画
    """
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0]
                                               for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()
