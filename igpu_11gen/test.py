import numpy as np

def process_pixels_row_major(block):
    """ Process a 4x4 pixel block in row-major order """
    processed_pixels = []
    for row in range(block.shape[0]):  # Row-wise iteration
        for col in range(block.shape[1]):  # Left-to-right in a row
            processed_pixels.append(block[row, col])
    return processed_pixels

def z_order_curve_4x4():
    """ Correct Z-order curve for a 4x4 block """
    return [
        (0,0), (0,1), (1,0), (1,1),
        (0,2), (0,3), (1,2), (1,3),
        (2,0), (2,1), (3,0), (3,1),
        (2,2), (2,3), (3,2), (3,3)
    ]

def process_pixels_z_order(block):
    """ Process a 4x4 pixel block in Z-order """
    z_order = z_order_curve_4x4()
    processed_pixels = [block[x, y] for x, y in z_order]
    return processed_pixels

# Example 4x4 block with pixel values (for simplicity, using numbers 1-16)
block = np.arange(1, 17).reshape(4, 4)
print("Original 4x4 Pixel Block:")
print(block)

# 8th Gen Processing (Row-Major)
row_major_pixels = process_pixels_row_major(block)
print("\n8th Gen Row-Major Pixel Order:")
print(row_major_pixels)

# 11th Gen Processing (Z-Order)
z_order_pixels = process_pixels_z_order(block)
print("\n11th Gen Z-Order Pixel Order:")
print(z_order_pixels)
