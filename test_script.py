from harness import aut
from mali import mali_compression
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':

    ''' verify the mali compression algo with aut.py ''' 

    random_list = [(random.randint(0, 128), random.randint(0, 64)) for _ in range(1000)]

    # dataset = np.random.randint(0, 10, (10, 4, 4, 8), dtype=np.uint8)
    cr = {}

    width, height = 512, 512

    r = np.tile(((np.linspace(0, 255, width) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (height, 1))  # Exponential Red scaling
    g = np.tile((np.sin(np.linspace(0, np.pi, height)) * 200).astype(np.uint8).reshape(height, 1), (1, width))  # Sinusoidal Green
    b = np.tile((np.linspace(0, 255, width) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (height, 1))  # Log-like Blue scaling
    a = np.tile((np.cos(np.linspace(0, np.pi, height)) * 255).astype(np.uint8).reshape(height, 1), (1, width))  # Cosine Alpha fading

    gradient_image = np.stack([r, g, b, a], axis=0)  # (4, 512, 512)

    # image = Image.fromarray(np.moveaxis(gradient_image, 0, -1), mode="RGBA")
    # image.save("harness/gradient_alpha_fixed.png")

    dataset = np.array([gradient_image])
    # TODO none needs to be replaced with something
    mali_gpu = aut.Aut(dataset, mali_compression.mali_compression, mali_compression.mali_decompression, None, name="Mali GPU") 
    mali_gpu.formal_validate()


