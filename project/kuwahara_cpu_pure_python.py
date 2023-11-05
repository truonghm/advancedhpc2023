import os
from math import sqrt
import time
from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def rgb_to_hsv_cpu_pp(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = float(mx - mn)
    h = 0
    if mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx
    v = mx
    return h, s, v

def get_mean_std_dev(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = sqrt(variance)
    return mean, std_dev

def get_window_values(image, x_start, y_start, x_end, y_end):
    values = []
    for y in range(max(0, y_start), min(y_end, len(image))):
        for x in range(max(0, x_start), min(x_end, len(image[0]))):
            values.append(image[y][x])
    return values


def kuwahara_filter_cpu_pp(image_array, window_size):
    height, width, _ = image_array.shape
    output_array = np.zeros_like(image_array)
    pad_size = window_size // 2

    for y in range(height):
        for x in range(width):
            min_std_dev = float('inf')
            min_mean = [0, 0, 0]

            for dy in range(-pad_size, pad_size + 1):
                for dx in range(-pad_size, pad_size + 1):
                    region_x_start = max(x + dx, 0)
                    region_y_start = max(y + dy, 0)
                    region_x_end = min(x + dx + window_size, width)
                    region_y_end = min(y + dy + window_size, height)

                    region = image_array[region_y_start:region_y_end, region_x_start:region_x_end]
                    if region.size == 0:
                        continue

                    v_values = [rgb_to_hsv_cpu_pp(*pixel)[2] for row in region for pixel in row]
                    mean, std_dev = get_mean_std_dev(v_values)
                    if std_dev < min_std_dev:
                        min_std_dev = std_dev
                        region_pixels = [pixel for row in region for pixel in row]
                        min_mean = [sum(p[i] for p in region_pixels) / len(region_pixels) for i in range(3)]

            output_array[y, x] = min_mean

    return output_array


def main(image_path: str, time_rult_path: str):
    
    img = imread(image_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    for w_size in window_sizes:
        print(f"Starting kuwahara CPU (pure python) with window size {w_size}")
        start = time.time()
        filtered_image_array = kuwahara_filter_cpu_pp(img, w_size)
        end = time.time()
        print(f"Time elapsed: {end-start}")
        with open(time_rult_path, 'a') as f:
            f.write(f'cpu_pp,{end - start},{w_size},None\n')
        imsave(f'outputs/filtered_image_cpu_pp_{w_size}.png', filtered_image_array)


if __name__ == '__main__':
	base_path = os.path.dirname(os.path.realpath(__file__))
	image_path = os.path.join(base_path, "original.png")
	main(image_path, os.path.join(base_path, "time_result.csv"))