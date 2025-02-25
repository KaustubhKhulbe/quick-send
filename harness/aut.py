import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pdb

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
    def __init__(self, dataset, compress, decompress, sample_d=uniform_sample, sample_i=None, name=""):
        self.dataset = dataset
        self.compress = compress
        self.decompress = decompress
        self.sample_d = sample_d
        self.name = name

    def validate(self, image):
        assert np.array_equal(self.decompress(self.compress(image)), image), f"AUT: {self.name} not lossless"

    def formal_validate(self):
        print(f"Begining formal validation of AUT: {self.name}")
        for i in tqdm(range(len(self.dataset))):
            self.validate(dataset[i])
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
def intel_iGPU_compress(image):
    # ensure image dimensions are multiples of 32
    width = math.ceil(float(image.shape[0]) / 32) * 32
    height = math.ceil(float(image.shape[1]) / 32) * 32

    image_pad = np.zeros((width, height))
    image_pad[:image.shape[0], :image.shape[1]] = image

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
            if max_res[i] == 0: compression_bits_required[i] = 0
            else: compression_bits_required[i] = math.ceil(np.log2(max_res[i]))


        compressable = True
        if np.sum(compression_bits_required) > 14: compressable = False

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
                diff = 0
                for channel in range(0, blk.shape[0]):
                    diff |= int((blk[channel][r][c] - mins[channel])) << int(channel*compression_bits_required[i])
                    residual_data[r * 8 + c] = diff

        cache_lines = None
        flag = 0

        if (compressable):
            cache_lines = np.zeros((64, 1), dtype=np.uint8)
            for i in range(6):
                    cache_lines[i] = (header >> (8 * (5 - i))) & 0xFF
            bit_pos = 48
            for num in residual_data:
                assert 0 <= num < (1 << 14)
                byte_idx = bit_pos // 8
                bit_offset = bit_pos % 8
                cache_lines[byte_idx] |= (int(num) >> (6 + bit_offset)) & 0xFF
                cache_lines[byte_idx + 1] = (int(num) >> (-2 + bit_offset)) & 0xFF if bit_offset >= 2 else (int(num) << (2 - bit_offset)) & 0xFF
                if bit_offset >= 2:
                    cache_lines[byte_idx + 2] |= (int(num) << (8 - (bit_offset - 2))) & 0xFF

                bit_pos += 14
            flag = 1

        else:
            cache_lines = blk.reshape(64, 2).astype(np.uint8)

        return cache_lines, flag

    gradient = np.linspace(0, 255, num=4*8, dtype=np.uint8).reshape(4, 8)
    z = np.zeros((4, 8), dtype=np.uint8)

    alpha_channel = np.full((4, 8), 255, dtype=np.uint8)  # (4,8)

    grayscale_rgba_planes = np.stack([gradient, z, z, alpha_channel], axis=0)  # (4,4,8)

    lines, flag = compress_4x8(grayscale_rgba_planes)
    print(lines.T)
    visual_rgb = np.moveaxis(grayscale_rgba_planes[:3], 0, -1)

    plt.imshow(visual_rgb, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


dataset = np.random.rand(10**3, 512, 512) * 255

intel_iGPU_compress(dataset[0])
