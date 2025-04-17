import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pdb
import random

cr = {}

def uniform_sample(dataset):
    return dataset[np.random.randint(0, dataset.shape[0])]

class Aut:
    '''
    dataset: numpy array where first dimension is an index to the image
    compress: compression algorithm: f(image) -> compressed
    decompress: decompression algorithm: f(compressed) -> image
    sample_d: sampling function for dataset (dataset) -> sample
    sample_i: sampling function for image (w, h, n) -> [indices]
    '''
    def __init__(self, dataset, compress, decompress, random_access, sample_d=uniform_sample, sample_i=None, name=""):
        self.dataset = dataset
        self.compress = compress
        self.decompress = decompress
        self.sample_d = sample_d
        self.name = name
        self.cr = None
        self.random_access = random_access

    def validate(self, image, height, width):
        assert np.array_equal(self.decompress(self.compress(image, height, width), height, width), image), f"AUT: {self.name} not lossless"

    def formal_validate(self):
        print(f"Begining formal validation of AUT: {self.name}")
        for i in tqdm(range(len(self.dataset))):
            self.validate(self.dataset[i], self.dataset.shape[2], self.dataset.shape[3])
        print(f"AUT ({self.name}) formally validated")

def identity_compress(image):
    return image

def identity_decompress(image):
    return image

'''
Encoding scheme:
Header: <r_skip> <g_skip> <b_skip> <a_skip> <r_pred> <g_pred> <b_pred> <a_pred> <r_bits> <g_bits> <b_bits> <a_bits>
Data Line: (rgba) (rgba) (rgba) ... each 14 bits wide for a total of 448 bits
Total: 48 bit header + 448 bit data line to get 496 bits or 62 byte cache line
'''
def intel_iGPU_compress(image, height, width):
    # Ensure the image is in (C, H, W) format where C=4 (RGBA)
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"

    # Ensure dimensions are multiples of 32
    height = math.ceil(image.shape[1] / 32) * 32
    width = math.ceil(image.shape[2] / 32) * 32

    # Pad image to (4, height, width)
    image_pad = np.zeros((4, height, width), dtype=np.uint8)
    image_pad[:, :image.shape[1], :image.shape[2]] = image

    compressed_data = []

    # 4 x 8 compression
    # returns a cache line and a flag
    def compress_4x8(blk):
        '''
        1. iterate through all 4x8 blocks
        2. calculate minimum X for all channels
        3. calculate all residuals
        4. calculate amount of encoding bits needed for block
        5. if less than 14 --> compress, otherwise use 2 cache lines
        6. return numpy array as bytes
        '''
        header = np.array((48,))
        mins = np.zeros((blk.shape[0], ), dtype=np.int64)
        for i in range(mins.shape[0]): mins[i] = np.min(blk[i,:,:])
        res = blk.copy()
        for i in range(mins.shape[0]): res[i] -= int(mins[i])



        max_res = np.zeros(mins.shape)
        for i in range(max_res.shape[0]): max_res[i] = int(np.max(res[i]))

        compression_bits_required = np.zeros(mins.shape)
        for i in range(compression_bits_required.shape[0]):
            if max_res[i] == 0:
                compression_bits_required[i] = 0
            else:
                log_val = np.log2(max_res[i])
                compression_bits_required[i] = math.ceil(log_val) + (log_val == math.floor(log_val))

        compressable = True
        if np.sum(compression_bits_required) > 14:
            compressable = False
            return blk.reshape(64, 2).astype(np.uint8), 0

        header = 0
        for i in range(max_res.shape[0]):
            skip_bit = 1 if compression_bits_required[i] == 0 else 0
            header |= skip_bit << (47 - i)

            prediction = mins[i]
            header |= ((0xFF) & prediction) << (48 - 4 - 8*(i+1))

            bits = compression_bits_required[i] - 1
            if bits < 0: bits = 0

            header |= (0x07 & int(bits)) << (48 - 4 - 32 - 3*(i+1))

        residual_data = np.zeros((32,), dtype=np.uint16)

        i = 0
        for c in range(0, 8):
            for r in range(0, 4):
                bit_string = ""

                for channel in range(blk.shape[0]):  # Iterate over channels
                    bit_width = int(compression_bits_required[channel])

                    if bit_width > 0:  # Skip channels with 0-bit storage
                        value = blk[channel][r][c] - mins[channel]  # Compute difference
                        bin_value = f"{value:0{bit_width}b}"  # Convert to binary

                        bit_string += bin_value  # Concatenate bits


                if len(bit_string) < 14: bit_string += "0" * (14 - len(bit_string))

                if bit_string:  # Ensure bit_string isn't empty
                    residual_data[r * 8 + c] = int(bit_string, 2)  # Convert to integer
                else:
                    residual_data[r * 8 + c] = 0  # Default value if all channels are 0 bits


        cache_lines = None
        flag = 0

        if (compressable):
            cache_lines = np.zeros((64, 1), dtype=np.uint8)
            for i in range(6):
                    cache_lines[i] = (header >> (8 * (5 - i))) & 0xFF

            bit_string = "".join(f"{num:014b}" for num in residual_data)  # 14-bit binary representation

            for i in range(0, len(bit_string), 8):
                byte_chunk = bit_string[i:i+8]
                cache_lines[6 + i // 8] = int(byte_chunk, 2)

            flag = 1

        else:
            cache_lines = blk.reshape(64, 2).astype(np.uint8)

        return cache_lines, flag


    for y in range(0, height, 4):
        for x in range(0, width, 8):
            blk = image_pad[:, y:y+4, x:x+8]  # Extract 4x8 block
            compressed_blk, flag = compress_4x8(blk)
            compressed_data.append((flag, compressed_blk))
            cr[(y // 4, x // 8)] = flag
    return compressed_data

    '''
    gradient = np.linspace(0, 64, num=4*8, dtype=np.uint8).reshape(4, 8)
    z = np.zeros((4, 8), dtype=np.uint8)

    alpha_channel = np.full((4, 8), 255, dtype=np.uint8)  # (4,8)

    grayscale_rgba_planes = np.stack([gradient + 20, gradient + 23, z + 100, alpha_channel], axis=0)  # (4,4,8)

    lines, flag = compress_4x8(grayscale_rgba_planes)
    image_data = intel_iGPU_decompress(lines, flag)
    visual_rgb = np.moveaxis(image_data[:3], 0, -1)
    visual_rgb_og = np.moveaxis(grayscale_rgba_planes[:3], 0, -1)

    # plt.imshow(visual_rgb_og, cmap='gray', vmin=0, vmax=255)
    plt.imshow(visual_rgb, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.show()
    '''

def intel_iGPU_decompress(compressed_data, height, width):
    image_reconstructed = np.zeros((4, height, width), dtype=np.uint8)

    def decompress_4x8(cache_lines, flag):
        if flag == 0: return cache_lines.reshape(4, 4, 8).astype(np.uint8)
        preds = np.zeros(4)
        header = 0
        for i in range(6):
            header |= int(cache_lines[i].item()) << (8 * (5 - i))

        skips = (header >> 44) & 0xF

        for i in range(4):
            preds[i] = (int(header) & (0xFF << (48 - 4 - 8*(i+1)))) >> (12 + 8*(3-i))

        bits_needed = np.zeros(4)
        for i in range(4):
            shift_amount = 48 - 4 - 32 - (3 * (i + 1))
            bits_needed[i] = ((header >> shift_amount) & 0x07) + 1
            if skips & (1 << (3 - i)): bits_needed[i] = 0

        bit_string = "".join(f"{int(cache_lines[i].item()):08b}" for i in range(6, 64))
        residual_data = np.array([int(bit_string[i:i+14], 2) for i in range(0, 32 * 14, 14)], dtype=np.uint16)

        image_data = np.zeros((4, 4, 8), dtype=np.uint8)

        for i in range(32):  # 32 pixels in a 4x8 grid
            base_value = int(residual_data[i])  # Extract stored 14-bit value
            bit_string = f"{base_value:014b}"  # Convert to 14-bit binary string

            r_bits = int(bits_needed[0].item())
            g_bits = int(bits_needed[1].item())
            b_bits = int(bits_needed[2].item())
            a_bits = int(bits_needed[3].item())

            shift = 0
            r_bin = bit_string[shift:shift + r_bits]
            shift += r_bits
            g_bin = bit_string[shift:shift + g_bits]
            shift += g_bits
            b_bin = bit_string[shift:shift + b_bits]
            shift += b_bits
            a_bin = bit_string[shift:shift + a_bits]

            r = int(r_bin, 2) if r_bin else 0
            g = int(g_bin, 2) if g_bin else 0
            b = int(b_bin, 2) if b_bin else 0
            a = int(a_bin, 2) if a_bin else 0

            r += int(preds[0])
            g += int(preds[1])
            b += int(preds[2])
            a += int(preds[3])

            row, col = divmod(i, 8)
            image_data[0][row][col] = r
            image_data[1][row][col] = g
            image_data[2][row][col] = b
            image_data[3][row][col] = a

        return image_data

    index = 0
    for y in range(0, height, 4):
        for x in range(0, width, 8):
            flag, compressed_blk = compressed_data[index]
            index += 1
            image_reconstructed[:, y:y+4, x:x+8] = decompress_4x8(compressed_blk, flag)

    return image_reconstructed

def intel_iGPU_random_access(x, y, cr):
    # print(len(cr))
    # print(x, y)
    if (y // 4, x // 8) not in cr:
        return 64
    flag = cr[(y // 4, x // 8)]
    if flag == 0: return 128
    else: return 64

def intel_iGPU_random_cr_benchmark(samples, cr):
    total = 0
    for s in samples:
        bytes_transferred = intel_iGPU_random_access(s[0], s[1], cr)
        total += bytes_transferred
    return total / len(samples)

def get_compressability_igpu_8gen(image, x, y):
    # Ensure the image is in (C, H, W) format where C=4 (RGBA)
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"
    global cr

    height = image.shape[1]
    width = image.shape[2]

    if cr == {}:
        dataset = np.array([image])
        # print(dataset.shape)
        igpu = Aut(dataset, intel_iGPU_compress, intel_iGPU_decompress, intel_iGPU_random_access, name="intel iGPU")
        igpu.compress(image, height, width)
    return intel_iGPU_random_access(x, y, cr)



# random_list = [(random.randint(0, 128), random.randint(0, 64)) for _ in range(1000)]

# # dataset = np.random.randint(0, 10, (10, 4, 4, 8), dtype=np.uint8)
# cr = {}

# from PIL import Image

# width, height = 512, 512

# r = np.tile(((np.linspace(0, 255, width) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (height, 1))  # Exponential Red scaling
# g = np.tile((np.sin(np.linspace(0, np.pi, height)) * 200).astype(np.uint8).reshape(height, 1), (1, width))  # Sinusoidal Green
# b = np.tile((np.linspace(0, 255, width) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (height, 1))  # Log-like Blue scaling
# a = np.tile((np.cos(np.linspace(0, np.pi, height)) * 255).astype(np.uint8).reshape(height, 1), (1, width))  # Cosine Alpha fading

# gradient_image = np.stack([r, g, b, a], axis=0)  # (4, 512, 512)

# image = Image.fromarray(np.moveaxis(gradient_image, 0, -1), mode="RGBA")
# image.save("gradient_alpha_fixed.png")

# dataset = np.array([gradient_image])
# igpu = Aut(dataset, intel_iGPU_compress, intel_iGPU_decompress, intel_iGPU_random_access, name="intel iGPU")

# igpu.compress(dataset[0], 512, 512)
# print(intel_iGPU_random_cr_benchmark(random_list, cr))

# image.show()
