import time
import numpy as np
import os
from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def rgb_to_hsv_cpu(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v

def kuwahara_filter_cpu(image_array, window_size):
    hsv_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=float)
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            hsv_array[i, j] = rgb_to_hsv_cpu(*image_array[i, j])

    output_array = np.zeros_like(image_array)

    pad_size = window_size // 2
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    padded_v = np.pad(hsv_array[:,:,2], ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    for i in range(pad_size, image_array.shape[0] + pad_size):
        for j in range(pad_size, image_array.shape[1] + pad_size):
            windows = [
                padded_v[i-pad_size:i+1, j-pad_size:j+1],
                padded_v[i-pad_size:i+1, j:j+pad_size+1],
                padded_v[i:i+pad_size+1, j-pad_size:j+1],
                padded_v[i:i+pad_size+1, j:j+pad_size+1]
            ]

            std_devs = [np.std(w) for w in windows]

            min_std_dev_index = np.argmin(std_devs)

            window_coords = [
                (slice(i-pad_size, i+1), slice(j-pad_size, j+1)),
                (slice(i-pad_size, i+1), slice(j, j+pad_size+1)),
                (slice(i, i+pad_size+1), slice(j-pad_size, j+1)),
                (slice(i, i+pad_size+1), slice(j, j+pad_size+1))
            ]
            mean_rgb = np.mean(padded_image[window_coords[min_std_dev_index]], axis=(0, 1))

            output_array[i-pad_size, j-pad_size] = mean_rgb

    return output_array

def main(image_path: str, time_rult_path: str):
    
    img = imread(image_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    for w_size in window_sizes:
        start = time.time()
        filtered_image_array = kuwahara_filter_cpu(img, w_size)
        end = time.time()
        # print(f'Kuwahara Filter with Window Size {w_size} took {end - start} seconds')
        # time_result.append({'type': 'cpu', 'time': end - start, 'window_size': w_size, 'note': None})
        with open(time_rult_path, 'a') as f:
            f.write(f'cpu,{end - start},{w_size},None\n')
        imsave(f'filtered_image_{w_size}.png', filtered_image_array)


if __name__ == '__main__':
	base_path = os.path.dirname(os.path.realpath(__file__))
	image_path = os.path.join(base_path, "original.png")
	main(image_path, os.path.join(base_path, "time_result.csv"))