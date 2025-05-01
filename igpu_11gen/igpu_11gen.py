import numpy as np
import random
import math
import matplotlib.pyplot as plt
from PIL import Image

# def sample(N, func, W, H):
#     if func == "random":
#         return [(random.randint(0, W-1), random.randint(0, H-1)) for _ in range(N)]
    
#     elif func == "gradient":
#         return [(int((i / N) * (W-1)), int((i / N) * (H-1))) for i in range(N)]
    
    
#     elif func == "periodic":
#         period = max(1, N // 10)
#         return [( (i % period) * W // period, (i % period) * H // period ) for i in range(N)]
    
#     elif func == "blurred":
#         center_x, center_y = W // 2, H // 2
#         return [(int(random.gauss(center_x, W//8)), int(random.gauss(center_y, H//8))) 
#                 for _ in range(N)]
    
#     elif func == "skew":
#         return [(int((random.random() ** 2) * (W-1)), int((random.random() ** 0.5) * (H-1))) 
#                 for _ in range(N)]
    
#     else:
#         raise ValueError(f"Unknown pattern: {func}")


# def check_uniform_2x2(block):
#     return np.all(block == block[0, 0, :])


# def compress_block(block):
#     min_values = np.zeros(4, dtype=np.uint8)
#     bit_lengths = np.zeros(4, dtype=int)
#     residuals = np.zeros_like(block)
    
#     # Process each channel (R, G, B, A) separately
#     for c in range(4):
#         min_values[c] = np.min(block[c, :, :])
#         residuals[c, :, :] = block[c, :, :] - min_values[c]
#         max_residual = np.max(residuals[c, :, :])
#         safe_max_residual = max(max_residual, 1)
#         bit_lengths[c] = int(np.ceil(np.log2(safe_max_residual + 1))) if safe_max_residual > 0 and not np.isinf(safe_max_residual) else 0
    
#     total_bits = np.sum(bit_lengths)
#     if total_bits > 14:
#         return {"flag": 0, "original_block": block.tolist()}
    
#     reserved_bits = np.zeros(8, dtype=int)
#     for i in range(0, 4, 2):
#         for j in range(0, 8, 2):
#             sub_block = block[:, i:i+2, j:j+2]
#             index = (i // 2) * 4 + (j // 2)
#             if check_uniform_2x2(sub_block):
#                 reserved_bits[index] = 1
    
#     # Only store bit lengths for R, G, and B channels (3 channels, 3*4 = 12 bits)
#     header = {
#         "color_transform": 0,
#         "reserved": reserved_bits.tolist(),
#         "min_values": min_values.tolist(),  # 4 channels * 8 bits = 32 bits
#         "bit_lengths": bit_lengths[:3].tolist()  # 3 channels * 4 bits = 12 bits
#     }
    
#     # Assert header length: 1 (color_transform) + 8 (reserved) + 12 (residuals for RGB) + 32 (min_values for RGBA) = 53 bits
#     if (1 + 8 + len(header["bit_lengths"]) * 4 + len(header["min_values"]) * 8) != 53:
#         raise AssertionError("Header length mismatch: Expected 53 bits")
#     if len(header["min_values"]) * 8 != 32:
#         raise AssertionError("Min value bits mismatch: Expected 32 bits")
#     if len(header["bit_lengths"]) * 4 != 12:
#         raise AssertionError("Residual bits mismatch: Expected 12 bits")
    
#     compressed_data = {"header": header, "residuals": residuals.tolist()}

    
#     return {"flag": 1, "compressed_data": compressed_data}


# def compress_image(image, H, W ):

#     assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"

#     # Ensure dimensions are multiples of 32
#     H = math.ceil(image.shape[1] / 32) * 32
#     W = math.ceil(image.shape[2] / 32) * 32

#     # Pad image to (4, height, width)
#     image_pad = np.zeros((4, H, W), dtype=np.uint8)
#     image_pad[:, :image.shape[1], :image.shape[2]] = image

#     compressed_blocks = []
    
    
#     for i in range(0, H, 4):
#         for j in range(0, W, 8):
#             block = image_pad[:, i:i+4, j:j+8]
#             compressed_block = compress_block(block)
#             compressed_blocks.append(compressed_block)
#             cr[i // 4, j // 8] = compressed_block["flag"]
#     return compressed_blocks 

cr={}
def check_uniform_2x2(block):
    return np.all(block == block[:, 0:1, 0:1])


def pack_residuals(residuals, bit_lengths):
    flat_bits = ""
    for c in range(3):  # Only RGB
        bits = bit_lengths[c]
        for val in residuals[c].flatten():
            flat_bits += format(int(val), f'0{bits}b')
    padded_len = (8 - len(flat_bits) % 8) % 8
    flat_bits += '0' * padded_len
    byte_array = bytearray(int(flat_bits[i:i + 8], 2) for i in range(0, len(flat_bits), 8))
    return byte_array, padded_len


def unpack_residuals(byte_array, bit_lengths, shape, pad_bits):
    flat_bits = ''.join(format(byte, '08b') for byte in byte_array)
    flat_bits = flat_bits[:-pad_bits] if pad_bits > 0 else flat_bits

    residuals = np.zeros((4, shape[1], shape[2]), dtype=np.uint8)
    idx = 0
    for c in range(3):
        bits = bit_lengths[c]
        for i in range(shape[1] * shape[2]):
            val = int(flat_bits[idx:idx+bits], 2)
            y, x = divmod(i, shape[2])
            residuals[c, y, x] = val
            idx += bits
    return residuals


def compress_block(block):
    min_values = np.min(block, axis=(1, 2)).astype(np.uint8)
    residuals = block.astype(np.int16) - min_values[:, None, None]
    bit_lengths = np.array([int(np.ceil(np.log2(np.max(residuals[c]) + 1))) if np.max(residuals[c]) > 0 else 1 for c in range(4)])

    reserved_bits = np.zeros(8, dtype=int)
    uniform_mask = np.zeros((4, 4, 8), dtype=bool)
    uniform_blocks = 0

    for i in range(0, 4, 2):
        for j in range(0, 8, 2):
            sub_block = block[:, i:i+2, j:j+2]
            index = (i // 2) * 4 + (j // 2)
            if check_uniform_2x2(sub_block):
                reserved_bits[index] = 1
                uniform_blocks += 1
                uniform_mask[:, i:i+2, j:j+2] = True

    effective_residuals = np.where(uniform_mask, 0, residuals).astype(np.uint8)
    a_residuals = effective_residuals[3].tolist()  # store A channel residuals separately

    max_bit_budget = 22 if uniform_blocks >= 4 else 14
    if np.sum(bit_lengths[:3]) > max_bit_budget:
        return {"flag": 0, "original_block": block.tolist()}

    packed_residuals, pad_bits = pack_residuals(effective_residuals, bit_lengths[:3])

    header = {
        "color_transform": 0,
        "reserved": reserved_bits.tolist(),
        "min_values": min_values.tolist(),
        "bit_lengths": bit_lengths[:3].tolist(),
        "pad_bits": pad_bits
    }

    compressed_data = {
        "header": header,
        "residuals_packed": list(packed_residuals),
        "a_residuals": a_residuals
    }

    return {"flag": 1, "compressed_data": compressed_data}


def compress_image(image, H, W):
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"

    # Ensure dimensions are multiples of 32
    H = math.ceil(image.shape[1] / 32) * 32
    W = math.ceil(image.shape[2] / 32) * 32

    # Pad image to (4, height, width)
    image_pad = np.zeros((4, H, W), dtype=np.uint8)
    image_pad[:, :image.shape[1], :image.shape[2]] = image
    compressed_blocks = []
    for i in range(0, H, 4):
        for j in range(0, W, 8):
            block = image_pad[:, i:i+4, j:j+8]
            compressed_block = compress_block(block)
            compressed_blocks.append(compressed_block)
            cr[i // 4, j // 8] = compressed_block["flag"]
    return compressed_blocks


def decompress_block(compressed_block):
    if compressed_block["flag"] == 0:
        return np.array(compressed_block["original_block"], dtype=np.uint8)

    header = compressed_block["compressed_data"]["header"]
    min_values = np.array(header["min_values"], dtype=np.uint8).reshape(4, 1, 1)
    bit_lengths = header["bit_lengths"]
    pad_bits = header["pad_bits"]
    shape = (4, 4, 8)

    packed = bytearray(compressed_block["compressed_data"]["residuals_packed"])
    residuals = unpack_residuals(packed, bit_lengths, shape, pad_bits)

    a_vals = np.array(compressed_block["compressed_data"]["a_residuals"], dtype=np.uint8).reshape(4, 8)
    residuals[3] = a_vals

    reserved = np.array(header["reserved"], dtype=np.uint8)
    reconstructed_block = np.zeros((4, 4, 8), dtype=np.uint8)

    for i in range(0, 4, 2):
        for j in range(0, 8, 2):
            index = (i // 2) * 4 + (j // 2)
            if reserved[index] == 1:
                for c in range(4):
                    reconstructed_block[c, i:i+2, j:j+2] = min_values[c, 0, 0]
            else:
                for c in range(4):
                    reconstructed_block[c, i:i+2, j:j+2] = residuals[c, i:i+2, j:j+2] + min_values[c, 0, 0]

    return reconstructed_block.astype(np.uint8)


def decompress_image(compressed_data, H, W):
    decompressed_image = np.zeros((4, H, W), dtype=np.uint8)
    index = 0
    for i in range(0, H, 4):
        for j in range(0, W, 8):
            decompressed_image[:, i:i+4, j:j+8] = decompress_block(compressed_data[index])
            index += 1
    return decompressed_image


# def test_compression():
#     H, W = 512, 512
#     r = np.tile(((np.linspace(0, 255, W) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (H, 1))
#     g = np.tile((np.sin(np.linspace(0, np.pi, H)) * 200).astype(np.uint8).reshape(H, 1), (1, W))
#     b = np.tile((np.linspace(0, 255, W) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (H, 1))
#     a = np.tile((np.cos(np.linspace(0, np.pi, H)) * 255).astype(np.uint8).reshape(H, 1), (1, W))
#     image = np.stack([r, g, b, a], axis=0)
#     img = Image.fromarray(image.transpose(1, 2, 0), mode="RGBA")
#     img.save("gradient_alpha_fixed.png")

#     compressed_data = compress_image(image, H, W)
#     decompressed_image = decompress_image(compressed_data, H, W)

#     compressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 1)
#     uncompressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 0)
#     total_blocks = len(compressed_data)

#     original_size = total_blocks * 128
#     compressed_size = (compressed_blocks_count * 64) + (uncompressed_blocks_count * 128)
#     compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

#     print(f"Total blocks: {total_blocks}")
#     print(f"Compressed blocks: {compressed_blocks_count}")
#     print(f"Uncompressed blocks: {uncompressed_blocks_count}")
#     print(f"Compression Ratio: {compression_ratio:.2f}")
#     print(f"Decompression Lossless: {np.array_equal(image, decompressed_image)}")


# test_compression()


def intel_iGPU_random_access(x, y, cr):
    # print(len(cr))
    # print(x, y)
    if (y // 4, x // 8) not in cr:
         return 64
    flag = cr[(y // 4, x // 8)]
    if flag == 0: return 128
    else: return 64

def get_compressability_igpu_11gen(image, x, y):
    # Ensure the image is in (C, H, W) format where C=4 (RGBA)
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"
    global cr

    H = image.shape[1]
    W = image.shape[2]

    if cr == {}:
    
    # print(dataset.shape)
        compress_image(image, H, W)
    
    return intel_iGPU_random_access(x, y, cr)

# def decompress_block(compressed_block):
#     if compressed_block["flag"] == 0:
#         return np.array(compressed_block["original_block"], dtype=np.uint8)
    
#     header = compressed_block["compressed_data"]["header"]
#     residuals = np.array(compressed_block["compressed_data"]["residuals"], dtype=np.uint8)
#     min_values = np.array(header["min_values"], dtype=np.uint8).reshape(4, 1, 1)
#     return residuals + min_values


# def decompress_image(compressed_data, H, W):
#     decompressed_image = np.zeros((4, H, W), dtype=np.uint8)
#     index = 0
#     for i in range(0, H, 4):
#         for j in range(0, W, 8):
#             decompressed_image[:, i:i+4, j:j+8] = decompress_block(compressed_data[index])
#             index += 1
#     return decompressed_image




def test_compression():
    H, W = 512, 512
    r = np.tile(((np.linspace(0, 255, W) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (H, 1))
    g = np.tile((np.sin(np.linspace(0, np.pi, H)) * 200).astype(np.uint8).reshape(H, 1), (1, W))
    b = np.tile((np.linspace(0, 255, W) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (H, 1))
    a = np.tile((np.cos(np.linspace(0, np.pi, H)) * 255).astype(np.uint8).reshape(H, 1), (1, W))
    image = np.stack([r, g, b, a], axis=0)
    img = Image.fromarray(image.transpose(1, 2, 0), mode="RGBA")
    img.save("gradient_alpha_fixed.png")
    

    # N=1000
    # points = sample(N, "random", W, H)
    # x= points[0]
    # y= points[1]
    compressed_data = compress_image(image, H, W)
    decompressed_image = decompress_image(compressed_data, H, W)
    
    compressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 1)
    uncompressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 0)
    total_blocks = len(compressed_data)
    
    original_size = total_blocks * 128
    compressed_size = (compressed_blocks_count * 64) + (uncompressed_blocks_count * 128)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    print(f"Total blocks: {total_blocks}")
    print(f"Compressed blocks: {compressed_blocks_count}")
    print(f"Uncompressed blocks: {uncompressed_blocks_count}")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Decompression Lossless: {np.array_equal(image, decompressed_image)}")

# test_compression()