import os
import glob
import itertools

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# r = np.tile(((np.linspace(0, 255, width) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (height, 1))  # Exponential Red scaling
# g = np.tile((np.sin(np.linspace(0, np.pi, height)) * 200).astype(np.uint8).reshape(height, 1), (1, width))  # Sinusoidal Green
# b = np.tile((np.linspace(0, 255, width) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (height, 1))  # Log-like Blue scaling
# a = np.tile((np.cos(np.linspace(0, np.pi, height)) * 255).astype(np.uint8).reshape(height, 1), (1, width))  # Cosine Alpha fading

# gradient_image = np.stack([r, g, b, a], axis=0)  # (4, 512, 512)

# image = Image.fromarray(np.moveaxis(gradient_image, 0, -1), mode="RGBA")
# image.save('../samples/gradient/gradient_alpha_fixed.png')


def create_gradient(width, height, angle, start_color, end_color):
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    theta = np.radians(angle)
    gradient = np.cos(theta) * X + np.sin(theta) * Y

    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

    gradient_image = np.zeros((height, width, 3))
    for i in range(3):
        gradient_image[:, :, i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    
    return gradient_image

def arr_to_image(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))

def generate_gradient_samples(base_path, count, width, height):
    samples = []

    for i in range(count):
        angle = np.random.uniform(0, 90)
        start_color = np.random.uniform((0, 0, 0), (1, 1, 1))
        end_color = np.random.uniform((0, 0, 0), (1, 1, 1))

        gradient = create_gradient(width, height, angle, start_color, end_color)
        samples.append(gradient)

        filename = os.path.join(base_path, f'{i+1}.png')
        arr_to_image(gradient).save(filename)

    return samples

def generate_blur_samples(base_path, source_path, s=5):
    exts = ['*.png', '*.jpg', '*.tiff']
    source_files = itertools.chain(*[glob.glob(os.path.join(source_path, ext)) for ext in exts])

    samples = []

    for source_file in source_files:
        source = np.array(Image.open(source_file).convert('RGB'))

        blur = gaussian_filter(source, sigma=(s, s, 0))
        samples.append(blur)

        filename = os.path.join(base_path, f'{os.path.basename(source_file)}_blur.png')
        arr_to_image(blur).save(filename)
    
    return samples


generate_gradient_samples('../samples/gradient/', 10, 128, 128)
generate_blur_samples('../samples/blur/', '../samples/aerials')
generate_blur_samples('../samples/blur/', '../samples/textures')
