from PIL import Image
from pylab import *

image = array(Image.open('demo.png').convert('L'))

imshow(image)

print('put 3 points')
x = ginput(3)
print(f'clicked point: {x[0]} {x[1]} {x[2]}')

show()