from PIL import Image
import numpy as np

image: np.ndarray = np.array(Image.open('demo.png').convert('L'))

image_re: np.ndarray = 255 - image
image_low_contrast: np.ndarray = (100.0 / 255) * image + 100
image_high_contrast: np.ndarray = 255.0 * (image / 255.0) ** 2

pil_image = Image.fromarray(image)
pil_image_re = Image.fromarray(image_re)
pil_image_lc = Image.fromarray(np.uint8(image_low_contrast))
pil_image_hc = Image.fromarray(np.uint8(image_high_contrast))

pil_image.show()
pil_image_re.show()
pil_image_lc.show()
pil_image_hc.show()
