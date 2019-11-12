from PIL import Image
import numpy as np
import imtools

image: np.ndarray = np.array(Image.open('demo.png').convert('L'))
image_after, cdf = imtools.histequal(image)

image_after = Image.fromarray(np.uint8(image_after))
image_after.show()

image_avg = imtools.compute_average(['demo.png', 'pil_image'])
image_avg = Image.fromarray(image_avg)
image_avg.show()