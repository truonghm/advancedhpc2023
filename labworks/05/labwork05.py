import math
from numba import cuda, float32
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.image import imread
import os

start = time.time()


blur_kernel = np.array(
    [
        [0, 0, 1, 2, 1, 0, 0],
        [0, 3, 13, 22, 13, 3, 0],
        [1, 13, 59, 97, 59, 13, 1],
        [2, 22, 97, 159, 97, 22, 2],
        [1, 13, 59, 97, 59, 13, 1],
        [0, 3, 13, 22, 13, 3, 0],
        [0, 0, 1, 2, 1, 0, 0],
    ]
)
blur_kernel = blur_kernel / blur_kernel.sum()


@cuda.jit
def blur_kernel_2d(src, dst, kernel):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        for c in range(3):
            conv_sum = 0
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if (
                        x + i >= 0
                        and x + i < src.shape[0]
                        and y + j >= 0
                        and y + j < src.shape[1]
                    ):
                        conv_sum += (
                            src[x + i, y + j, c] * kernel[i + 3, j + 3]
                        )
            dst[x, y, c] = conv_sum



def gaussian_blur_gpu_2d(
    img,
    dir_to_save: str,
    block_size: int,
    kernel_func,
    blur_kernel,
    show_result: bool = False,
):
    h, w, c = img.shape
    pixel_count = h * w
    grid_size_x = math.ceil(h / block_size[0])
    grid_size_y = math.ceil(w / block_size[1])
    grid_size = (grid_size_x, grid_size_y)
    rgb = np.ascontiguousarray(img[..., :3])

    devSrc = cuda.to_device(rgb)
    devDst = cuda.device_array((h, w, 3), dtype=np.float32)

    kernel_func[grid_size, block_size](devSrc, devDst, blur_kernel)

    hostDst = devDst.copy_to_host()
    blurred_img = hostDst.reshape(h, w, 3)
    # Display the blurred image
    fig = plt.figure()
    plt.imshow(blurred_img)
    if show_result:
        plt.axis("off")
        plt.show()
    plt.imsave(
        os.path.join(dir_to_save, f"blurred_gpu_block_size_{block_size}.png"),
        blurred_img,
    )
    plt.close(fig)


if __name__ == "__main__":
    N_TRIALS = 10
    print(f"Running each test {N_TRIALS} times")
    base_path = os.path.dirname(os.path.realpath(__file__))
    img = imread(os.path.join(base_path, "original.png"))

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
            gaussian_blur_gpu_2d(
                img,
                base_path,
                block_size,
                blur_kernel_2d,
                blur_kernel,
                show_result=False,
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
    plt.xticks([i for i in range(len(time_results))], block_sizes_to_test[1:])
    # plt.show()
    fig.savefig(os.path.join(base_path, "block_size_vs_time.png"), bbox_inches="tight")
