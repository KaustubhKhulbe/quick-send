import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pdb
import random

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
            self.validate(dataset[i], dataset.shape[2], dataset.shape[3])
        print(f"AUT ({self.name}) formally validated")


def identity_compress(image):
    return image

def identity_decompress(image):
    return image

'''
Encoding scheme:
Header: <r_skip> <g_skip> <b_skip> <a_skip> <r_pred> <g_pred> <b_pred> <a_pred> <r_bits> <g_bits> <b_bits> <a_bits>
Data Line: (rgba) (rgba) (rgba) ... each 14 bits wide for a total of 448 bits
Total: 53 bit header + 448 bit data line to get 496 bits or 62 byte cache line
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
        """
        1. Iterate through all 4x8 blocks (RGBA)
        2. Convert to Z-order (2x2 sub-blocks)
        3. Compute min values per channel
        4. Compute residuals
        5. Compute required bit-width per channel
        6. Encode the new 53-bit header
        7. Check compression feasibility (≤14-bit per pixel)
        8. Store compressed residuals in cachelines
        """
        
        header = 0
        mins = np.zeros(blk.shape[0], dtype=np.int64)
        for i in range(mins.shape[0]):
            mins[i] = np.min(blk[i, :, :])
        
        res = blk.copy()
        for i in range(mins.shape[0]):
            res[i] -= int(mins[i])
        
        max_res = np.zeros(mins.shape)
        for i in range(max_res.shape[0]):
            max_res[i] = int(np.max(res[i]))
        
        compression_bits_required = np.zeros(mins.shape)
        for i in range(compression_bits_required.shape[0]):
            if max_res[i] == 0:
                compression_bits_required[i] = 0
            else:
                log_val = np.log2(max_res[i])
                compression_bits_required[i] = math.ceil(log_val) + (log_val == math.floor(log_val))
        
        # Ensure total bits for residuals do not exceed 14 per pixel
        if np.sum(compression_bits_required[:3]) > 14:
            return blk.reshape(64, 2).astype(np.uint8), 0
        
        # Construct header
        for i in range(3):  # RGB channels (Alpha inferred)
            bits = compression_bits_required[i] - 1 if compression_bits_required[i] > 0 else 0
            header |= (0x0F & int(bits)) << (53 - 4 * (i + 1))
        
        # Store the 32-bit prediction
        for i in range(3):
            header |= (mins[i] & 0xFF) << (53 - 12 - 8 * (i + 1))
        
        # Convert header into bytes
        cache_lines = np.zeros((64,), dtype=np.uint8)
        for i in range(7):  # 53-bit header fits in 7 bytes
            cache_lines[i] = (header >> (8 * (6 - i))) & 0xFF
        
        # Pack residual data
        residual_data = np.zeros((32,), dtype=np.uint16)
        bit_string = ""
        for c in range(8):
            for r in range(4):
                pixel_bits = ""
                for channel in range(3):
                    bit_width = int(compression_bits_required[channel])
                    if bit_width > 0:
                        value = blk[channel][r][c] - mins[channel]
                        pixel_bits += f"{value:0{bit_width}b}"
                if len(pixel_bits) < 14:
                    pixel_bits += "0" * (14 - len(pixel_bits))
                bit_string += pixel_bits
        
        # Ensure bit_string is correctly packed into cache_lines
        for i in range(0, len(bit_string), 8):
            if 7 + i // 8 < 64:
                cache_lines[7 + i // 8] = int(bit_string[i:i+8], 2)
        
        return cache_lines, 1

    for y in range(0, height, 4):
        for x in range(0, width, 8):
            blk = image_pad[:, y:y+4, x:x+8]  # Extract 4x8 block
            compressed_blk, flag = compress_4x8(blk)
            compressed_data.append((flag, compressed_blk))
            cr[(y // 4, x // 8)] = flag
    return compressed_data


def intel_iGPU_decompress(compressed_data, height, width):
    image_reconstructed = np.zeros((4, height, width), dtype=np.uint8)

    def decompress_4x8(cache_lines, flag):
        if flag == 0:  
            return cache_lines.reshape(4, 4, 8).astype(np.uint8)  # Uncompressed case

        header = 0
        for i in range(7):  # 53-bit header
            header |= int(cache_lines[i].item()) << (8 * (6 - i))

        color_transform_flag = (header >> 52) & 0x1  # 1-bit color flag
        skips = (header >> 44) & 0xFF  # 8 reserved bits (ignored)
        
        bits_needed = np.zeros(4, dtype=int)
        min_values = np.zeros(4, dtype=int)

        # Extract RGB ℓ(x) values (4-bit each)
        for i in range(3):
            shift_amount = 40 - (i * 4)
            bits_needed[i] = ((header >> shift_amount) & 0x0F) + 1  # ℓ(R), ℓ(G), ℓ(B)
        
        # Extract Min RGB Values (8-bit each)
        for i in range(4):
            shift_amount = 24 - (i * 8)
            min_values[i] = (header >> shift_amount) & 0xFF  # Min values for R, G, B, A

        # Read compressed residuals
        bit_string = "".join(f"{int(cache_lines[i].item()):08b}" for i in range(7, 64))
        residual_data = np.array([int(bit_string[i:i+14], 2) for i in range(0, 32 * 14, 14)], dtype=np.uint16)

        image_data = np.zeros((4, 4, 8), dtype=np.uint8)
        z_order = [(0,0), (0,1), (1,0), (1,1)]  # 2×2 Z-order pattern

        for i in range(32):  # 32 pixels in a 4x8 grid
            base_value = int(residual_data[i])  
            bit_string = f"{base_value:014b}"

            r_bits, g_bits, b_bits = bits_needed[:3]
            a_bits = 14 - (r_bits + g_bits + b_bits)  # Use leftover bits for alpha

            shift = 0
            r = int(bit_string[shift:shift + r_bits], 2) if r_bits else 0
            shift += r_bits
            g = int(bit_string[shift:shift + g_bits], 2) if g_bits else 0
            shift += g_bits
            b = int(bit_string[shift:shift + b_bits], 2) if b_bits else 0
            shift += b_bits
            a = int(bit_string[shift:shift + a_bits], 2) if a_bits else 0

            # Add min values back
            r += min_values[0]
            g += min_values[1]
            b += min_values[2]
            a += min_values[3]

            # Convert i to Z-order 2×2 blocks
            block_x = (i % 8) // 2 * 2
            block_y = (i // 8) * 2
            sub_index = (i % 8) % 2 + ((i // 8) % 2) * 2
            dy, dx = z_order[sub_index]

            image_data[0][block_y + dy][block_x + dx] = r
            image_data[1][block_y + dy][block_x + dx] = g
            image_data[2][block_y + dy][block_x + dx] = b
            image_data[3][block_y + dy][block_x + dx] = a

        return image_data

    index = 0
    for y in range(0, height, 4):
        for x in range(0, width, 8):
            flag, compressed_blk = compressed_data[index]
            index += 1
            image_reconstructed[:, y:y+4, x:x+8] = decompress_4x8(compressed_blk, flag)

    return image_reconstructed



def intel_iGPU_random_access(x, y, cr):
    flag = cr[(y // 4, x // 8)]
    if flag == 0: return 64
    else: return 32

def intel_iGPU_random_cr_benchmark(samples, cr):
    total = 0
    for s in samples:
        bytes_transferred = intel_iGPU_random_access(s[0], s[1], cr)
        total += bytes_transferred
    return total / len(samples)

random_list = [(random.randint(0, 128), random.randint(0, 64)) for _ in range(1000)]

# dataset = np.random.randint(0, 10, (10, 4, 4, 8), dtype=np.uint8)
cr = {}

from PIL import Image

width, height = 512, 512

r = np.tile(((np.linspace(0, 255, width) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (height, 1))  # Exponential Red scaling
g = np.tile((np.sin(np.linspace(0, np.pi, height)) * 200).astype(np.uint8).reshape(height, 1), (1, width))  # Sinusoidal Green
b = np.tile((np.linspace(0, 255, width) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (height, 1))  # Log-like Blue scaling
a = np.tile((np.cos(np.linspace(0, np.pi, height)) * 255).astype(np.uint8).reshape(height, 1), (1, width))  # Cosine Alpha fading

gradient_image = np.stack([r, g, b, a], axis=0)  # (4, 512, 512)

image = Image.fromarray(np.moveaxis(gradient_image, 0, -1), mode="RGBA")
image.save("gradient_alpha_fixed.png")

dataset = np.array([gradient_image])
igpu = Aut(dataset, intel_iGPU_compress, intel_iGPU_decompress, intel_iGPU_random_access, name="intel iGPU")

igpu.compress(dataset[0], 512, 512)
print(intel_iGPU_random_cr_benchmark(random_list, cr))

image.show()
