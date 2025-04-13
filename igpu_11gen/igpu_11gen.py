import numpy as np
import random
import math
import matplotlib.pyplot as plt
from PIL import Image

def sample(N, func, H, W):
    if func == "random":
        return [(random.randint(0, W-8), random.randint(0, H-4)) for _ in range(N)]
    
    elif func == "gradient":
        return [(int((i / N) * (W-8)), int((i / N) * (H-4))) for i in range(N)]
    
    elif func == "periodic":
        period = max(1, N // 10)
        return [( (i % period) * W // period, (i % period) * H // period ) for i in range(N)]
    
    elif func == "blurred":
        center_x, center_y = W // 2, H // 2
        return [(int(random.gauss(center_x, W//8)), int(random.gauss(center_y, H//8))) 
                for _ in range(N)]
    
    elif func == "skew":
        return [(int((random.random() ** 2) * (W-8)), int((random.random() ** 0.5) * (H-4))) 
                for _ in range(N)]

    
    else:
        raise ValueError(f"Unknown pattern: {func}")



def check_uniform_2x2(block):
    return np.all(block == block[0, 0, :])


def compress_block(block):
    min_values = np.zeros(4, dtype=np.uint8)
    bit_lengths = np.zeros(4, dtype=int)
    residuals = np.zeros_like(block)
    
    # Process each channel (R, G, B, A) separately
    for c in range(4):
        min_values[c] = np.min(block[c, :, :])
        residuals[c, :, :] = block[c, :, :] - min_values[c]
        max_residual = np.max(residuals[c, :, :])
        safe_max_residual = max(max_residual, 1)
        bit_lengths[c] = int(np.ceil(np.log2(safe_max_residual + 1))) if safe_max_residual > 0 and not np.isinf(safe_max_residual) else 0
    
    total_bits = np.sum(bit_lengths)
    if total_bits > 14:
        return {"flag": 0, "original_block": block.tolist()}
    
    reserved_bits = np.zeros(8, dtype=int)
    for i in range(0, 4, 2):
        for j in range(0, 8, 2):
            sub_block = block[:, i:i+2, j:j+2]
            index = (i // 2) * 4 + (j // 2)
            if check_uniform_2x2(sub_block):
                reserved_bits[index] = 1
    
    # Only store bit lengths for R, G, and B channels (3 channels, 3*4 = 12 bits)
    header = {
        "color_transform": 0,
        "reserved": reserved_bits.tolist(),
        "min_values": min_values.tolist(),  # 4 channels * 8 bits = 32 bits
        "bit_lengths": bit_lengths[:3].tolist()  # 3 channels * 4 bits = 12 bits
    }
    
    # Assert header length: 1 (color_transform) + 8 (reserved) + 12 (residuals for RGB) + 32 (min_values for RGBA) = 53 bits
    if (1 + 8 + len(header["bit_lengths"]) * 4 + len(header["min_values"]) * 8) != 53:
        raise AssertionError("Header length mismatch: Expected 53 bits")
    if len(header["min_values"]) * 8 != 32:
        raise AssertionError("Min value bits mismatch: Expected 32 bits")
    if len(header["bit_lengths"]) * 4 != 12:
        raise AssertionError("Residual bits mismatch: Expected 12 bits")
    
    compressed_data = {"header": header, "residuals": residuals.tolist()}
    return {"flag": 1, "compressed_data": compressed_data}


def compress_image(image, H, W, points):
    compressed_blocks = []
    for x, y in points:
        # Ensure that the block stays within bounds
        # if x + 4 <= H and y + 8 <= W:
        block = image[:, x:x+4, y:y+8]
        compressed_blocks.append(compress_block(block))
    return compressed_blocks


def decompress_block(compressed_block):
    if compressed_block["flag"] == 0:
        return np.array(compressed_block["original_block"], dtype=np.uint8)
    
    header = compressed_block["compressed_data"]["header"]
    residuals = np.array(compressed_block["compressed_data"]["residuals"], dtype=np.uint8)
    min_values = np.array(header["min_values"], dtype=np.uint8).reshape(4, 1, 1)
    return residuals + min_values


def decompress_image(compressed_data, H, W, points):
    decompressed_image = np.zeros((4, H, W), dtype=np.uint8)
    index = 0
    for x, y in points:
        # Ensure that the block stays within bounds
        # if x + 4 <= H and y + 8 <= W:
        decompressed_image[:, x:x+4, y:y+8] = decompress_block(compressed_data[index])
        index += 1
    return decompressed_image


def test_compression():
    H, W = 512, 512
    r = np.tile(((np.linspace(0, 255, W) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (H, 1))
    g = np.tile((np.sin(np.linspace(0, np.pi, H)) * 200).astype(np.uint8).reshape(H, 1), (1, W))
    b = np.tile((np.linspace(0, 255, W) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (H, 1))
    a = np.tile((np.cos(np.linspace(0, np.pi, H)) * 255).astype(np.uint8).reshape(H, 1), (1, W))
    image = np.stack([r, g, b, a], axis=0)
    img = Image.fromarray(image.transpose(1, 2, 0), mode="RGBA")
    img.save("gradient_alpha_fixed.png")
    
    N = 1000
    points = sample(N, "skew", H, W)
    print(len(points))
    compressed_data = compress_image(image, H, W, points)
    decompressed_image = decompress_image(compressed_data, H, W, points)
    
    # Count compressed and uncompressed blocks
    compressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 1)
    
    uncompressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 0)
    total_blocks = (compressed_blocks_count+uncompressed_blocks_count)
    
    # Original block size (128 bytes per block)
    original_size = total_blocks * 128
    
    # Compressed block size (64 bytes for compressed, 128 bytes for uncompressed)
    compressed_size = (compressed_blocks_count * 64) + (uncompressed_blocks_count * 128)
    
    # Compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    print(f"Total blocks: {total_blocks}")
    print(f"Compressed blocks: {compressed_blocks_count}")
    print(f"Uncompressed blocks: {uncompressed_blocks_count}")
    print(f"Compression Ratio: {compression_ratio:.2f}")

# test_compression()
