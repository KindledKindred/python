from PIL import Image  # pillow から Image オブジェクトをインポート
import io  # ネットワーク関連
import urllib.request  # URL 処理関連

# 画像をLoremPicsumから読み込んで保存

pil_image = Image.open(io.BytesIO(urllib.request.urlopen(
    "https://picsum.photos/600/600").read()))
pil_image.save("pil_image.png")

# サムネイルの作成

# pil_image.thumbnail((128,128))

box = (0, 0, 300, 300)

#pil_image.paste(pil_thumnail, box)

# 切り抜いて回転させて貼り付け

region = pil_image.crop(box)
region.save("region.png")

region = region.transpose(Image.ROTATE_180)

pil_image.paste(region, box)
pil_image.save("pil_pasted.png")

region = region.transpose(Image.ROTATE_180)
#pil_roteted = pil_image.paste(region, box)
# if pil_roteted is not None:
#  pil_roteted.save("csacma.png")
