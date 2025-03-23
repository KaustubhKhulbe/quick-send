import numpy as np

def compress_block(block):
    """
    Compresses a single 4x8 block of RGBA values using the 11th Gen Intel GPU compression algorithm.
    Assumes all blocks are non-uniform.
    """
    # Step 1: Compute predictions (minimum values per channel)
    min_values = np.min(block, axis=(0, 1))  # Shape: (4,) for RGBA
    
    # Step 2: Compute residuals
    residuals = block - min_values  # Shape: (4,8,4)
    
    # Step 3: Compute bit-lengths for residuals
    max_residuals = np.max(residuals, axis=(0,1))
    safe_max_residuals = np.where(max_residuals > 0, max_residuals, 1)  # Avoid divide by zero
    bit_lengths = np.ceil(np.log2(safe_max_residuals + 1)).astype(int)
    total_bits = np.sum(bit_lengths)
    
    # Step 4: Check 14-bit per pixel budget
    if total_bits > 14:
        return {"flag": 0, "original_block": block.tolist()}  # Store original block if compression fails
    
    # Step 5: Construct compressed block (header + residuals)
    header = {
        "color_transform": 0,  # 1-bit color transform flag (default 0)
        "reserved": 0,  # 8 reserved bits (default 0)
        "min_values": min_values.tolist(),
        "bit_lengths": bit_lengths.tolist(),
    }
    compressed_data = {
        "header": header,
        "residuals": residuals.tolist()
    }
    
    return {"flag": 1, "compressed_data": compressed_data}

def compress_image(image):
    """
    Compresses an image (H, W, 4) using 4x8 blocks.
    """
    H, W, C = image.shape
    assert C == 4, "Image must have 4 channels (RGBA)"
    
    compressed_blocks = []
    for i in range(0, H, 4):
        for j in range(0, W, 8):
            block = image[i:i+4, j:j+8]  # Extract 4x8 block
            compressed_block = compress_block(block)
            compressed_blocks.append(compressed_block)
    
    return compressed_blocks

def decompress_block(compressed_block):
    """
    Decompresses a single 4x8 block.
    """
    if compressed_block["flag"] == 0:
        return np.array(compressed_block["original_block"], dtype=np.uint8)  # Restore original block
    
    header = compressed_block["compressed_data"]["header"]
    residuals = np.array(compressed_block["compressed_data"]["residuals"], dtype=np.uint8)
    min_values = np.array(header["min_values"], dtype=np.uint8)
    
    # Reconstruct original block
    decompressed_block = residuals + min_values
    return decompressed_block

def decompress_image(compressed_data, H, W):
    """
    Decompresses the image from compressed data.
    """
    decompressed_image = np.zeros((H, W, 4), dtype=np.uint8)
    index = 0
    for i in range(0, H, 4):
        for j in range(0, W, 8):
            decompressed_block = decompress_block(compressed_data[index])
            decompressed_image[i:i+4, j:j+8] = decompressed_block
            index += 1
    
    return decompressed_image

# Test function with a gradient RGBA image
# def test_compression():
#     H, W = 512, 512  # Image size
#     image = np.zeros((H, W, 4), dtype=np.uint8)
#     for i in range(H):
#         for j in range(W):
#             image[i, j] = [i % 256, j % 256, (i + j) % 256, 255]  # Gradient pattern with full alpha
    
#     compressed_data = compress_image(image)
#     decompressed_image = decompress_image(compressed_data, H, W)
    
#     compressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 1)
#     uncompressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 0)
#     total_blocks = len(compressed_data)
    
#     original_size = total_blocks * 128  # 128 bytes per block (original)
#     compressed_size = (compressed_blocks_count * 64) + (uncompressed_blocks_count * 128)  # 64 bytes for compressed, 128 for uncompressed
#     compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
#     print(f"Total blocks: {total_blocks}")
#     print(f"Compressed blocks: {compressed_blocks_count}")
#     print(f"Uncompressed blocks: {uncompressed_blocks_count}")
#     print(f"Compression Ratio: {compression_ratio:.2f}")
    
#     # Check if decompressed image matches original
#     is_lossless = np.array_equal(image, decompressed_image)
#     print(f"Decompression Lossless: {is_lossless}")
    
from PIL import Image

def test_compression():
    H, W = 512, 512  # Image size

    # Generate skewed test image
    r = np.tile(((np.linspace(0, 255, W) ** 6 / (255 / 6)) ** 0.5).astype(np.uint8), (H, 1))  # Exponential Red scaling
    g = np.tile((np.sin(np.linspace(0, np.pi, H)) * 200).astype(np.uint8).reshape(H, 1), (1, W))  # Sinusoidal Green
    b = np.tile((np.linspace(0, 255, W) ** 0.5 / 255 ** 0.5 * 255).astype(np.uint8), (H, 1))  # Log-like Blue scaling
    a = np.tile((np.cos(np.linspace(0, np.pi, H)) * 255).astype(np.uint8).reshape(H, 1), (1, W))  # Cosine Alpha fading

    image = np.stack([r, g, b, a], axis=-1)  # Shape: (512, 512, 4)

    # Save test image
    img = Image.fromarray(image, mode="RGBA")
    img.save("gradient_alpha_fixed.png")
    img.show()
    # Compression and decompression
    compressed_data = compress_image(image)
    decompressed_image = decompress_image(compressed_data, H, W)

    compressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 1)
    uncompressed_blocks_count = sum(1 for block in compressed_data if block["flag"] == 0)
    total_blocks = len(compressed_data)

    original_size = total_blocks * 128  # 128 bytes per block (original)
    compressed_size = (compressed_blocks_count * 64) + (uncompressed_blocks_count * 128)  # 64 bytes for compressed, 128 for uncompressed
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    print(f"Total blocks: {total_blocks}")
    print(f"Compressed blocks: {compressed_blocks_count}")
    print(f"Uncompressed blocks: {uncompressed_blocks_count}")
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Check if decompressed image matches original
    is_lossless = np.array_equal(image, decompressed_image)
    print(f"Decompression Lossless: {is_lossless}")

# Run test
test_compression()

# Run test
# test_compression()
