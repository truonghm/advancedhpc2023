import os
import time

from matplotlib.image import imread, imsave

from kuwahara_cpu_numpy import kuwahara_filter_cpu
from kuwahara_cpy_pure_python import kuwahara_filter_cpu_pp

from enum import Enum

class FuncType(str, Enum):
	CPU = "cpu"
	GPU = "gpu"

def main(image_path: str, time_rult_path: str, func: callable, func_type: str = FuncType.CPU):
    img = imread(image_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]

    # for GPU init
    if func_type == FuncType.GPU:
        print("GPU Init")
        _ = func(img, 3)

    for w_size in window_sizes:
        print(f"Starting {func.__name__} ({func_type}) with window size {w_size}")
        start = time.time()
        filtered_image_array = func(img, w_size)
        end = time.time()
        print(f'Kuwahara Filter with Window Size {w_size} took {end - start} seconds')
        with open(time_rult_path, "a") as f:
            f.write(f"cpu,{end - start},{w_size},None\n")
        imsave(f"filtered_image_{w_size}.png", filtered_image_array)


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    # base_path = ""
    image_path = os.path.join(base_path, "original.png")
    # main(os.path.join(base_path, "time_result.csv"), kuwahara_filter_cpu, FuncType.CPU)
    main(image_path, os.path.join(base_path, "time_result.csv"), kuwahara_filter_cpu_pp, FuncType.CPU)
