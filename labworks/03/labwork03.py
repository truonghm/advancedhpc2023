from numba import cuda, float32
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.image import imread
import os

start = time.time()


@cuda.jit
def grayscale_kernel(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.float32((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g
    # dst[tidx] = g


def to_grayscale_gpu(
    img, dir_to_save: str, block_size: int, kernel, show_result: bool = False
):
    h, w, c = img.shape
    pixel_count = h * w
    grid_size = pixel_count // block_size + 1
    rgb = np.ascontiguousarray(img[..., :3].reshape(pixel_count, 3))

    devSrc = cuda.to_device(rgb)
    devDst = cuda.device_array((pixel_count, 3), dtype=np.float32)

    kernel[grid_size, block_size](devSrc, devDst)

    hostDst = devDst.copy_to_host()
    grayscale_img = hostDst.reshape(h, w, 3)
    # Display the grayscale image
    fig = plt.figure()
    plt.imshow(grayscale_img, cmap="gray")
    if show_result:
        plt.axis("off")
        plt.show()
    plt.imsave(
        os.path.join(dir_to_save, f"grayscale_gpu_block_size_{block_size}.png"),
        grayscale_img,
        cmap="gray",
    )
    plt.close(fig)


if __name__ == "__main__":
    N_TRIALS = 10
    print(f"Running each test {N_TRIALS} times")
    base_path = os.path.dirname(os.path.realpath(__file__))
    img = imread(os.path.join(base_path, "original.png"))

    block_sizes_to_test = [32, 32, 64, 128, 256, 512, 1024]
    time_results = []

    for idx, block_size in enumerate(block_sizes_to_test):
        total_time = 0
        for i in range(N_TRIALS):
            start = time.time()
            to_grayscale_gpu(
                img, base_path, block_size, grayscale_kernel, show_result=False
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
    fig.savefig(os.path.join(base_path, "block_size_vs_time.png"), bbox_inches="tight")