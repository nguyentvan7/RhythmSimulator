import cv2
from datetime import datetime
import numpy as np
import itertools
import collections
import sys
import glob
import os
import csv
from PIL import Image
from os import path

def s(x0):
    return int(float(x0))

def encoder(image, regions, debug=False, write=False):
    # Used for rate.
    prev_image = None
    bitmask = []
    encoded_pixels = []
    row_offsets = []
    images = 0
    
    pixels = 0
    input_image = cv2.imread(image)
    height, width, channels = input_image.shape
    output_image = np.zeros((height, width, channels), np.uint8)
    # Check for full frame.
    if regions == None:
        input_regions = [[0.0, height, width, 0.0, 1 , 1]]
    else:
        # Read region file.
        # Sort by x0.
        input_regions = sorted(list(csv.reader(open(regions), delimiter=',')), key=lambda region:int(float(region[0])))

    # Iterate over input image.
    for row in range(height):
        rowmask = []
        row_offsets.append(pixels)
        for col in range(width//2):
            pixelmask = 0b11
            # Iterate over regions.
            for region in input_regions:
                # Need to flip y. Original 0,0 is bottom left. New 0,0 is top left.
                x0 = int(float(region[0]))
                y0 = height-int(float(region[1]))
                x1 = int(float(region[2]))
                # Minor optimization.
                if col*2 < x0:
                    break
                y1 = height-int(float(region[3]))
                stride = int(region[4])
                skip = int(region[5])-1
                # Check if pixel is in region
                if x0 <= col*2 <= x1 and y0 <= row <= y1:
                    if pixelmask != 0b00:
                        if (col*2 & stride) > 0:
                            pixelmask |= 0b10
                        else:
                            pixelmask &= 0b01
                        if skip:
                            pixelmask |= 0b01
                        else:
                            pixelmask &= 0b10
                        if pixelmask == 0b11:
                            pixelmask = 0b10
                    if pixelmask == 0b00:
                        pixels += 2
                        encoded_pixels.append(input_image[row, col*2])
                        encoded_pixels.append(input_image[row, col*2+1])
                        break
            rowmask.append(pixelmask)
        bitmask.append(rowmask)

    fill = 0
    while (pixels + fill) % width != 0:
        encoded_pixels.append(0)
        encoded_pixels.append(0)
        fill += 2

    # Save image
    encoded_image = np.reshape(np.array(encoded_pixels, dtype=np.uint8), (-(-pixels//width), 640, 3))
    if debug:
        cv2.imshow('img', encoded_image)
        cv2.waitKey(0)
    if write:
        output_name = image.split('/')[1]
        encoded_image.save(output_folder_name + '/encoded/' + output_name)
    return encoded_image, bitmask, row_offsets
    
def decoder(encoded_images, bitmasks, row_offsets, height):
    encoded_height, width, channels = encoded_image.shape
    encoded_image = np.zeros((encoded_height, width, channels), np.uint8)
    encoded_px = 0
    for row in range(height):
        for col in range(width//2):
            pixelmask = bitmask.pop(0)
            if pixelmask == 0b00:
                # Regional pixel, copy to encoded and output image.
                encoded_image[encoded_px // width, encoded_px % width] = input_image[row, col*2]
                output_image[row, col*2] = encoded_image[encoded_px // width, encoded_px % width]
                encoded_px += 1
                encoded_image[encoded_px // width, encoded_px % width] = input_image[row, col*2+1]
                output_image[row, col*2+1] = encoded_image[encoded_px // width, encoded_px % width]
                encoded_px += 1
            elif pixelmask == 0b10:
                # Stride, copy from left encoded px.
                output_image[row, col*2] = encoded_image[(encoded_px-2) // width, (encoded_px-2) % width]
                output_image[row, col*2+1] = encoded_image[(encoded_px-1) // width, (encoded_px-1) % width]
            elif pixelmask == 0b01:
                # Skip, copy from previous frame.
                output_image[row, col*2] = prev_image[row, col*2]
                output_image[row, col*2+1] = prev_image[row, col*2+1]
            else:
                # No pixel, black.
                output_image[row, col*2] = 0
                output_image[row, col*2+1] = 0

    prev_image = output_image

    cv2.imwrite(output_folder_name + '/' + output_name, output_image)
    cv2.imwrite(output_folder_name + '/encoded/' + output_name, encoded_image)
    images += 1
    print(int(images/filecount*100), "% done (", images, "/", filecount, ")", sep="")

def get_px_index(encoded_image, bitmask, row_offset, col):
    count = 0
    for i in range(col):
        if bitmask[i] == '3':
            count += 1

    encoded_height, width, channels = encoded_image.shape
    regional_px_cnt = row_offset + count
    row = regional_px_cnt / width
    col = regional_px_cnt % width

    return row, col
    
