import math
import os
import time
from matplotlib.image import imread, imsave
from numba import cuda, float32
import numpy as np

@cuda.jit
def rgb_to_hsv_gpu_shared(src, dst):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    shared_rgb = cuda.shared.array(shape=(8, 8, 3), dtype=float32)

    if x < src.shape[0] and y < src.shape[1]:

        for i in range(3):
            shared_rgb[tx, ty, i] = src[x, y, i] / 255.0
        cuda.syncthreads()

        if ty < 8 and tx < 8:
            r, g, b = shared_rgb[tx, ty, 0], shared_rgb[tx, ty, 1], shared_rgb[tx, ty, 2]
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            df = max_val - min_val
            if max_val == min_val:
                h = 0
            elif max_val == r:
                h = np.float32((60 * ((g - b) / df) + 360) % 360)
            elif max_val == g:
                h = np.float32((60 * ((b - r) / df) + 120) % 360)
            elif max_val == b:
                h = np.float32((60 * ((r - g) / df) + 240) % 360)
            if max_val == 0:
                s = 0
            else:
                s = df / max_val
            v = max_val
            # Save the results back to global memory
            dst[x, y, 0] = h
            dst[x, y, 1] = s
            dst[x, y, 2] = v
        cuda.syncthreads()


@cuda.jit
def kuwahara_filter_gpu_shared(input_image, output_image, window_size):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # block_size = 8
    radius = window_size // 2
    diameter = window_size
    shared_img = cuda.shared.array(shape=(26, 26, 3), dtype=float32)

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            global_x = x + j
            global_y = y + i
            if (0 <= global_x < input_image.shape[0]) and (0 <= global_y < input_image.shape[1]):
                shared_img[tx + radius + i, ty + radius + j, 0] = input_image[global_x, global_y, 0]
                shared_img[tx + radius + i, ty + radius + j, 1] = input_image[global_x, global_y, 1]
                shared_img[tx + radius + i, ty + radius + j, 2] = input_image[global_x, global_y, 2]

    cuda.syncthreads()

    if ty >= radius and ty < 26 - radius and tx >= radius and tx < 26 - radius:
        if x >= radius and y >= radius and x < input_image.shape[0] - radius and y < input_image.shape[1] - radius:
            min_std_dev = 1e10
            min_mean = cuda.local.array(3, float32)

            for i in range(4):
                sum_r = sum_g = sum_b = sum_v = sum_v2 = count = 0
                x_start, x_end, y_start, y_end = 0, 0, 0, 0
                if i == 0:
                    x_start, x_end, y_start, y_end = tx, tx + diameter, ty, ty + diameter
                elif i == 1:
                    x_start, x_end, y_start, y_end = tx, tx + diameter, ty - diameter, ty
                elif i == 2:
                    x_start, x_end, y_start, y_end = tx - diameter, tx, ty, ty + diameter
                else:  # i == 3
                    x_start, x_end, y_start, y_end = tx - diameter, tx, ty - diameter, ty

                for dy in range(y_start, y_end):
                    for dx in range(x_start, x_end):
                        pixel_r = shared_img[dx, dy, 0]
                        pixel_g = shared_img[dx, dy, 1]
                        pixel_b = shared_img[dx, dy, 2]
                        pixel_v = (pixel_r + pixel_g + pixel_b) / 3
                        sum_r += pixel_r
                        sum_g += pixel_g
                        sum_b += pixel_b
                        sum_v += pixel_v
                        sum_v2 += pixel_v * pixel_v
                        count += 1

                mean_v = sum_v / count
                std_dev_v = math.sqrt(sum_v2 / count - mean_v * mean_v)

                if std_dev_v < min_std_dev:
                    min_std_dev = std_dev_v
                    min_mean[0] = sum_r / count
                    min_mean[1] = sum_g / count
                    min_mean[2] = sum_b / count

            output_image[x, y, 0] = min_mean[0]
            output_image[x, y, 1] = min_mean[1]
            output_image[x, y, 2] = min_mean[2]
    cuda.syncthreads()


def run(image, block_size, w_size):
    image_gpu = cuda.to_device(image)
    hsv_gpu = cuda.device_array_like(image_gpu)
    output_image_gpu = cuda.device_array_like(image_gpu)

    h, w, c = image.shape

    grid_size_x = math.ceil(h / block_size[0])
    grid_size_y = math.ceil(w / block_size[1])
    grid_size = (grid_size_x, grid_size_y)

    rgb_to_hsv_gpu_shared[grid_size, block_size](image_gpu, hsv_gpu)

    kuwahara_filter_gpu_shared[grid_size, block_size](
        image_gpu, output_image_gpu, w_size
    )
    output_image = output_image_gpu.copy_to_host()
    return output_image

def main(image_path: str, time_result_path: str):
    image = imread(image_path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = np.ascontiguousarray(image[:, :, :3])
        
    image_gpu = cuda.to_device(image)
    hsv_gpu = cuda.device_array_like(image_gpu)
    output_image_gpu = cuda.device_array_like(image_gpu)

    block_sizes = [(8, 8)]
    window_sizes = [3, 5, 7, 9]
    h, w, c = image.shape

    # init
    run(image, block_sizes[0], window_sizes[0])
    for block_size in block_sizes:
        for w_size in window_sizes:
            start = time.time()
            grid_size_x = math.ceil(h / block_size[0])
            grid_size_y = math.ceil(w / block_size[1])
            grid_size = (grid_size_x, grid_size_y)

            rgb_to_hsv_gpu_shared[grid_size, block_size](image_gpu, hsv_gpu)

            kuwahara_filter_gpu_shared[grid_size, block_size](
                image_gpu, output_image_gpu, w_size
            )
            output_image = output_image_gpu.copy_to_host()

            end = time.time()

            with open(time_result_path, 'a') as f:
                f.write(f'gpu_shared,{end - start},{w_size},"{block_size}"\n')
            img_dir = os.path.dirname(image_path)
            output_name = os.path.join(img_dir, "outputs", f'filtered_image_gpu_shared_{block_size}_{w_size}.png')
            imsave(output_name, output_image)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(base_path, "original.png")
    main(image_path, os.path.join(base_path, "time_result.csv"))