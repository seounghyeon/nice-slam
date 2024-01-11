import numpy as np
import cv2
import torch


def img_pre(image1in):

    if image1in is None or image1in.numel() == 0:
        # print("\nTHIS IS NONE IN IMAGE1IN no previous image saved up\n\n")
        return None, None, None, None
    # detach input images and change them from tensor to cv2 format
    ############################################
    # Detach the tensor from the GPU
    image1in_cpu = image1in.cpu()

    # Convert to NumPy arrays
    np_img1 = image1in_cpu.numpy()

    # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
    np_img1 = np.clip(np_img1, 0, 1)
    # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
    # print("Shape of np_img2:", np_img2.shape)
    if np_img1.shape[0] == 3:
        np_img1 = np.transpose(np_img1, (1, 2, 0))
    # If color values are in [0,1], scale to [0,255]
    if np_img1.max() <= 1.0:
        np_img1 = (np_img1 * 255).astype(np.uint8)
    # Save the images
    # image1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    print("Size of image1:", image1.shape)
    return image1