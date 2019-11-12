import cv2

if __name__ == "__main__":
    images = ['./pil_image.png', './pil_pasted.png', './region.png']

    one = cv2.imread(images[0])
    two = cv2.imread(images[1])
    three = cv2.imread(images[2])

    stitcher = cv2.createStitcher(False)
    result = stitcher.stitch((one, two, three))
    cv2.imwrite('./image_mosicing.png', result[1])
    if result[0] == 0:
        print("success")
    elif result[0] == 1:
        print("failure")
