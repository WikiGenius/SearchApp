# Author: Muhammed Elyamani
# Date: 03/02/2023
# GitHub: https://github.com/WikiGenius

import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def preprocess(img, new_shape=(640, 640)):
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, ratio, dwdh = letterbox(image, new_shape, auto=False)
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    image /= 255
    return image,  ratio, dwdh

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def create_rounded_img(img, border_radius = 50):
    # get the width and height of the image
    h, w = img.shape[:2]

    # create a mask with same height and width of the original image 
    mask = np.zeros((h,w), np.uint8)

    # Draw a rounded rectangle
    rect_color = (255, 255, 255)
    rect_top_left = (0, 0)
    rect_bottom_right = (w, h)
    cv2.rectangle(mask, (rect_top_left[0] + border_radius, rect_top_left[1]), (rect_bottom_right[0] - border_radius, rect_bottom_right[1]), rect_color, -1)
    cv2.rectangle(mask, (rect_top_left[0], rect_top_left[1] + border_radius), (rect_bottom_right[0], rect_bottom_right[1] - border_radius), rect_color, -1)
    cv2.circle(mask, (rect_top_left[0] + border_radius, rect_top_left[1] + border_radius), border_radius, rect_color, -1)
    cv2.circle(mask, (rect_bottom_right[0] - border_radius, rect_top_left[1] + border_radius), border_radius, rect_color, -1)
    cv2.circle(mask, (rect_top_left[0] + border_radius, rect_bottom_right[1] - border_radius), border_radius, rect_color, -1)
    cv2.circle(mask, (rect_bottom_right[0] - border_radius, rect_bottom_right[1] - border_radius), border_radius, rect_color, -1)
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # Create a black background image with the same height and width of the original image
    background = np.zeros((h, w, 3), np.uint8)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Apply the inverted mask to the background image
    background = cv2.bitwise_and(background, background, mask=mask_inv)

    # Combine the masked image and the background using addWeighted
    alpha = 1
    beta = 0
    gamma = 0
    rounded_img = cv2.addWeighted(masked_image, alpha, background, beta, gamma)
    return rounded_img

