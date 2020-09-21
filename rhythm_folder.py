import cv2
from datetime import datetime
import numpy as np
import itertools
import collections
import sys
import glob
import os
import csv
from os import path

def s(x0):
    return int(float(x0))

def main():
    # Parse command line arguments.
    input_folder_name = ""
    output_folder_name = ""
    region_folder_name = ""
    height = 0
    width = 0
    output = True
    
    arg_iter = iter(sys.argv)
    # Skip script name.
    next(arg_iter)
    for argument in arg_iter:
        if argument == "--help" or argument == "-h":
            print_usage()
            exit(1)
        else:
            if argument == "--input" or argument == "-i":
                argument = next(arg_iter)
                input_folder_name = argument
            elif argument == "--output" or argument == "-o":
                argument = next(arg_iter)
                output_folder_name = argument
            elif argument == "--region" or argument == "-r":
                argument = next(arg_iter)
                region_folder_name = argument
            elif argument == "--nooutput":
                argument = next(arg_iter)
                height = int(argument)
                argument = next(arg_iter)
                width = int(argument)
                output = False
            else:
                print("Invalid argument. Use -h or --help flag to print usage.")
                exit(1)
    if (input_folder_name == "" and output) or output_folder_name == "" or region_folder_name == "":
        print("Invalid argument(s). Use -h or --help flag to print usage.")
        exit(1)

    # Check input folder.
    if output:
        if not path.exists(input_folder_name):
            print("Input folder does not exist.")
            exit(1)
        if not path.isdir(input_folder_name):
            print("Input is not a folder.")
            exit(1)
    # Create output folder.
    if not path.exists(output_folder_name):
        os.mkdir(output_folder_name)
        print("Created output folder.")
    if not path.exists(output_folder_name + "/encoded"):
        os.mkdir(output_folder_name + "/encoded")
    if not path.exists(output_folder_name + "/stats"):
        os.mkdir(output_folder_name + "/stats")
    # Check region folder.
    if not path.exists(region_folder_name):
        print("Region folder does not exist.")
        exit(1)
    if not path.isdir(region_folder_name):
        print("Region is not a folder.")
        exit(1)
    filecount = len(os.listdir(input_folder_name))
    print("Starting processing on", filecount, "frames.")
    start = datetime.now()

    regionlist = glob.glob(region_folder_name + "/*")
    regionlist = sorted(regionlist, reverse=True)
    if output:
        imagelist = glob.glob(input_folder_name + "/*")
        imagelist = sorted(imagelist, reverse=True)
    else:
        imagelist = []
    imageregionlist = list(itertools.zip_longest(imagelist, regionlist))
    imageregionlist.reverse()
    # Used for rate.
    prev_image = None
    bitmasks = collections.deque(maxlen=4)
    encoded_images = collections.deque(maxlen=4)
    row_offsets = collections.deque(maxlen=4)
    images = 0
    zero_p = np.zeros(3, np.uint8)
    total_pixel_touches = 0
    total_bitmask_touches = 0
    total_row_offset_touches = 0
    total_bits = 0
    
    # Iterate over all images.
    for image, regions in imageregionlist:
        pixel_touches = 0
        bitmask_touches = 0
        row_offset_touches = 0
        pixels = 0
        bitmask = []
        encoded_pixels = []
        row_offset = []
        if output:
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
            # Check if file is empty
            if not input_regions:
                input_regions = [[0.0, height, width, 0.0, 1 , 1]]

        # Iterate over input image.
        for row in range(height):
            rowmask = []
            row_offset.append(pixels)
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
                            if output:
                                encoded_pixels.append(input_image[row, col*2])
                                encoded_pixels.append(input_image[row, col*2+1])
                            break
                rowmask.append(pixelmask)
            bitmask.append(rowmask)
        bitmasks.appendleft(bitmask)
        row_offsets.appendleft(row_offset)
        # Fill last row.
        while (pixels) % width != 0:
            if output:
                encoded_pixels.append(zero_p)
                encoded_pixels.append(zero_p)
            pixels += 2

        # Convert list to image.
        pixel_touches += pixels
        bitmask_touches += pixels/2
        row_offset_touches += height
        if output:
            encoded_image = np.reshape(np.array(encoded_pixels, dtype=np.uint8), (-(-pixels//width), width, channels))
            encoded_images.appendleft(encoded_image)

        # Decode image.
        for row in range(height):
            # Assume we cache the row_offset for each of the 4 frames at each row.
            row_offset_touches += 4
            # Assume we cache the bitmask for the row for each of the 4 frames at each row.
            bitmask_touches += 4*width//2
            for col in range(width//2):
                pixelmask = bitmask[row][col]
                if pixelmask == 0b01:
                    # Skip, check previous frames, most recent first, but skipping current.
                    for frame in range(1, len(bitmasks)):
                        if bitmasks[frame][row][col] == 0b00 or bitmasks[frame][row][col] == 0b10:
                            if output:
                                r, c = get_px_index(width, bitmasks[frame][row], row_offsets[frame][row], col)
                                output_image[row, col*2] = encoded_images[frame][r][c]
                                output_image[row, col*2+1] = encoded_images[frame][r][c+1]
                            pixel_touches += 2
                            break
                elif pixelmask == 0b00 or pixelmask == 0b10:
                    # Regional or strided pixel.
                    if output:
                        r, c = get_px_index(width, bitmask[row], row_offset[row], col)
                        output_image[row, col*2] = encoded_image[r][c]
                        output_image[row, col*2+1] = encoded_image[r][c+1]
                    pixel_touches += 2
                elif pixelmask == 0b11:
                    # Non-regional, make a black pixel.
                    if output:
                        output_image[row, col*2] = 0
                        output_image[row, col*2+1] = 0
                            
                        
        # Save image
        output_name = str(images)
        if output:
            output_name = image.split('/')[1]
            cv2.imwrite(output_folder_name + '/' + output_name, output_image)
            cv2.imwrite(output_folder_name + '/encoded/' + output_name, encoded_image)
        images += 1
        print(int(images/filecount*100), "% done (", images, "/", filecount, ")", sep="")
        # Stat calculation
        current_bits = pixel_touches*24 + bitmask_touches*2 + row_offset_touches*24
        print(round(current_bits/8000000, 2), "MB estimated with", format(pixel_touches, ","), "pixel touches,", format(int(bitmask_touches), ","), "bitmask touches, and", format(row_offset_touches, ","), "row offset touches for this image.")
        with open(output_folder_name + '/stats/' + output_name + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow((round(current_bits/8000000, 2), pixel_touches, int(bitmask_touches), row_offset_touches))
        # Total stat calculation
        total_bits += current_bits
        total_pixel_touches += pixel_touches
        total_bitmask_touches += bitmask_touches
        total_row_offset_touches += row_offset_touches
        print(round(total_bits/8000000, 2), "MB estimated with", format(total_pixel_touches, ","), "pixel touches,", format(int(total_bitmask_touches), ","), "bitmask touches, and", format(total_row_offset_touches, ","), "row offset touches in total.")
        print()

    with open(output_folder_name + '/stats/total.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow((round(total_bits/8000000, 2), total_pixel_touches, int(total_bitmask_touches), total_row_offset_touches))
    print("Completed in ", datetime.now()-start, "! Check ./", output_folder_name, " for decoded and encoded frames.", sep="")
    return

def get_px_index(width, rowmask, row_offset, col):
    count = 0
    for i in range(col):
        if rowmask[i] == 0b00:
            # Regional pixel.
            count += 2

    regional_px_cnt = row_offset + count
    row = regional_px_cnt // width
    col = regional_px_cnt % width

    return row, col
                
def print_usage():
    print("""\
Usage: rhythm_simulator.py OPTIONS
Simulates rhythm on the given input image and saves to given output image.
    -h, --help\t\tPrints this dialog and exits.
    -i, --input\t\tSpecifies the input folder, all images will be read in this folder.
    -o, --output\tSpecifies the output folder. The folder will be created if it doesn't exist.
    -e, --excel\t\tSpecifies the excel file with regions.""")
    return

if __name__ == "__main__":
    main()
