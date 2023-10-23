from numba import cuda, float32
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.image import imread
import os
import math

start = time.time()


@cuda.jit
def binary_kernel(src, dst, threshold):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        g = np.float32((src[x, y, 0] + src[x, y, 1] + src[x, y, 2]) / 3)
        if g < threshold:
            dst[x, y, 0] = dst[x, y, 1] = dst[x, y, 2] = 0
        else:
            dst[x, y, 0] = dst[x, y, 1] = dst[x, y, 2] = 1


def to_binary_gpu_2d(
    img, dir_to_save: str, block_size: int, kernel, threshold, show_result: bool = False
):
    h, w, c = img.shape
    pixel_count = h * w
    grid_size_x = math.ceil(h / block_size[0])
    grid_size_y = math.ceil(w / block_size[1])
    grid_size = (grid_size_x, grid_size_y)
    rgb = np.ascontiguousarray(img[..., :3])

    devSrc = cuda.to_device(rgb)
    devDst = cuda.device_array((h, w, 3), dtype=np.float32)

    kernel[grid_size, block_size](devSrc, devDst, threshold)

    hostDst = devDst.copy_to_host()
    grayscale_img = hostDst.reshape(h, w, 3)
    # Display the grayscale image
    fig = plt.figure()
    plt.imshow(grayscale_img, cmap="gray")
    if show_result:
        plt.axis("off")
        plt.show()
    plt.imsave(
        os.path.join(dir_to_save, f"binarization_gpu_block_size_{block_size}.png"),
        grayscale_img,
        cmap="gray",
    )
    plt.close(fig)


if __name__ == "__main__":
    N_TRIALS = 10
    threshold = 0.4
    print(f"Running each test {N_TRIALS} times")
    base_path = os.path.dirname(os.path.realpath(__file__))
    img = imread(os.path.join(base_path, "grayscale_input.png"))

    block_sizes_to_test = [
        (2, 2),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
    ]
    time_results = []

    for idx, block_size in enumerate(block_sizes_to_test):
        total_time = 0
        for i in range(N_TRIALS):
            start = time.time()
            to_binary_gpu_2d(
                img, base_path, block_size, binary_kernel, threshold, show_result=False
            )
            end = time.time()
            total_time += end - start
        if idx != 0:
            avg_time = total_time / N_TRIALS
            time_results.append(avg_time)
            print(f"Block size: {block_size}, time: {avg_time} seconds")

    fig = plt.figure()
    plt.plot(time_results)
    plt.xlabel("Block size")
    plt.ylabel("Time (seconds)")
    plt.xticks(
        [i for i in range(len(time_results))], block_sizes_to_test[1:]
    )
    # plt.show()
    fig.savefig(os.path.join(base_path, "binarization_block_size_vs_time.png"), bbox_inches="tight")
