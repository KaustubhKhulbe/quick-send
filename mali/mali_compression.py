import numpy as np 
import random

num_channels = 4
CHAN_Y = 0
CHAN_U = 1
CHAN_V = 2
CHAN_A = 3
CHAN_R = 0
CHAN_G = 1
CHAN_B = 2
ROOT = 0
def PARENT(node):
    return ( ((node) - 1) / 4 )
def CHILDOF(node, i):
    return ( 4*(node) + (i) + 1)

def QUAD_TO_PANE_X(quad):
    if(quad & 1):
        return 1 
    else:
        return 0

def QUAD_TO_PANE_Y(quad):
    if(quad & 2):
        return 1
    else:
        return 0

YUVA_CHANNEL_BITS = [8, 9, 9, 8]

# TODO bit length assertions
def yuva_to_rgba(yuva_pixel):
    yuva_pixel = tuple(int(x) for x in yuva_pixel)
    g = int((515 + 4*yuva_pixel[CHAN_Y] - yuva_pixel[CHAN_U] - yuva_pixel[CHAN_V]) / 4)
    r = yuva_pixel[CHAN_U] + g - 256
    b = yuva_pixel[CHAN_V] + g - 256
    a = yuva_pixel[CHAN_A]
    return (r, g, b, a)

def rgba_to_yuva(rgba_pixel):
    rgba_pixel = tuple(int(x) for x in rgba_pixel)
    u = rgba_pixel[CHAN_R] - rgba_pixel[CHAN_G] + 256
    v = rgba_pixel[CHAN_B] - rgba_pixel[CHAN_G] + 256
    y = int(np.ceil((4 * rgba_pixel[CHAN_G] - 515 + u + v) / 4))
    a = rgba_pixel[CHAN_A]
    return (y, u, v, a)

'''
    Notes: 
    * The length of the pane is stored in a header. Each pane is 6 bits in the header (0 is a weird case, 1 is used to represented uncompressed)
    * Pane is 4x4 pixels of 4 channels for a total of 64 total bytes 
    * Bitcount tree is 2 levels while val_tree is 3 levels. These are per channel
    * Bitcount tree is assembled first then the val_tree which depends on the bit count tree is assembled 
    * bitcount tree
        * 4 bits for the root value
        * 2 bits for each value of the quad 
    * val tree
        * root is channel bits for YUVA (8, 9, 9, 8)
        * root of bitcount tree used for the quads. 2 bits used for the zero quad, then root bitcount tree bits are used 
        * leaf pixels uses the root + child of bitcount tree bits. 2 bits used for the zero pixel, the total bits used for leaf pixel value. 
    * Indexing follows the pattern 
        1 2 
        3 4 
'''


'''
    Compresses a pane for mali compression 
    Input should be 4 channel x 4x4 pixel pane 
    returns (header (size in bytes), bitlength_tree, val_tree, child_zero_of_root, (children_zero of quads))
'''
def compress_pane(pane):
    val_tree = [] 
    bitlength_tree = []
    child_zero_of_root = []     # these are for the vals
    child_zeros_of_quads = [] # these are for the vals
    header_bits = 0             # maintain as bits for now and convert to bytes later

    # construct trees for each channel 
    for chan in range(num_channels):

        # Construct val_tree 
        channel_val_tree = np.zeros(21) 

        # Construct quad and qquad values 
        quad_vals = np.zeros(4)
        quad_child_zeros = np.zeros(4)
        for quad in range(4):
            quad_idx = CHILDOF(ROOT, quad) 

            # Get all the qquad pixels
            quad_pixels = np.array([ 
                pane[chan][2 * QUAD_TO_PANE_Y(quad) + QUAD_TO_PANE_Y(0)][2 * QUAD_TO_PANE_X(quad) + QUAD_TO_PANE_X(0)], 
                pane[chan][2 * QUAD_TO_PANE_Y(quad) + QUAD_TO_PANE_Y(1)][2 * QUAD_TO_PANE_X(quad) + QUAD_TO_PANE_X(1)], 
                pane[chan][2 * QUAD_TO_PANE_Y(quad) + QUAD_TO_PANE_Y(2)][2 * QUAD_TO_PANE_X(quad) + QUAD_TO_PANE_X(2)], 
                pane[chan][2 * QUAD_TO_PANE_Y(quad) + QUAD_TO_PANE_Y(3)][2 * QUAD_TO_PANE_X(quad) + QUAD_TO_PANE_X(3)]
            ])

            # generate quad val 
            quad_val = np.min(quad_pixels) 
            quad_child_zeros[quad] = np.argmin(quad_pixels) # determines the child_zero for that quad 
            quad_vals[quad] = quad_val 
            channel_val_tree[quad_idx] = quad_val 

            # fill in qquad vals 
            for qquad in range(4):
                qquad_idx = CHILDOF(quad_idx, qquad) 
                channel_val_tree[qquad_idx] = quad_pixels[qquad] - quad_val 
        
        # Calculate root val
        root_val = np.min(quad_vals)
        channel_val_tree[ROOT] = root_val 
        root_child_zero = np.argmin(quad_vals)
        for quad in range(4):
            quad_idx = CHILDOF(ROOT, quad) 
            channel_val_tree[quad_idx] -= root_val # update the tree 
            quad_vals[quad] -= root_val # update quad_vals for later 

        # Add root val to the header 
        header_bits += YUVA_CHANNEL_BITS[chan] 

        # Construct bit length tree 
        # TODO this is slightly modified from what hovav found 
        chan_bitlength_tree = np.zeros(5)

        # Calculate root bitcount 
        possible_root_bit_length = np.array([int(quad_val).bit_length() for quad_val in quad_vals])
        root_bit_length = np.max(possible_root_bit_length)
        chan_bitlength_tree[ROOT] = root_bit_length
        assert(int(root_bit_length).bit_length() <= 4)
        header_bits += 4 
        if(root_bit_length >= 2):
            header_bits += 3 * root_bit_length + 2   # optimization with zero child 
        else:
            header_bits += 4 * root_bit_length

        # Bitcount tree quads 
        for quad in range(4):
            quad_idx = CHILDOF(ROOT, quad) 

            # get qquad bit lengths 
            possible_quad_bit_lengths = np.array([int(channel_val_tree[CHILDOF(CHILDOF(ROOT, quad), qquad)]).bit_length() for qquad in range(4)])
            quad_bit_length = np.max(possible_quad_bit_lengths)
            bitlength_delta = quad_bit_length - root_bit_length 
            if(bitlength_delta < -2):
                bitlength_delta = -2 
            elif(bitlength_delta > 1): # TODO unsure about this case, I think if this happens we disable compression 
                # print("ERROR, THIS SHOULDN'T HAPPEN") 
                # exit(1)
                pass
            chan_bitlength_tree[quad_idx] = bitlength_delta 
            header_bits += 2 
            if(quad_bit_length >= 2):
                header_bits += quad_bit_length * 3 + 2
            else:
                header_bits += 4 * quad_bit_length

        # Channel trees to overall struct
        val_tree.append(channel_val_tree)
        bitlength_tree.append(chan_bitlength_tree)
        child_zero_of_root.append(root_child_zero)
        child_zeros_of_quads.append(quad_child_zeros)
    return (int(np.ceil(header_bits / 8)), bitlength_tree, val_tree, child_zero_of_root, child_zeros_of_quads) 

'''
    Decompresses a compressed input format of the pane 
    input: (header, bitlength_tree, val_tree, child_zero_of_root, child_zeros_of_quads)
'''
def decompress_pane(compressed_data):
    header, bitlength_tree, val_tree, child_zero_of_root, child_zeros_of_quads = compressed_data

    decompressed_yuva_pane = np.zeros((4, 4, 4)) 

    # Generally you would need to read the bitlength tree and go from there but we just need the val_tree for each channel
    for chan in range(4):
        for quad in range(4):
            quad_idx = CHILDOF(ROOT, quad) 
            for qquad in range(4): 
                qquad_idx = CHILDOF(quad_idx, qquad)
                val = val_tree[chan][ROOT] + val_tree[chan][quad_idx] + val_tree[chan][qquad_idx] 
                decompressed_yuva_pane[chan][2*QUAD_TO_PANE_Y(quad) + QUAD_TO_PANE_Y(qquad)][2*QUAD_TO_PANE_X(quad) + QUAD_TO_PANE_X(qquad)] = val 

    return decompressed_yuva_pane


'''
    Compress a single pane and returns compressed bytes
'''
def mali_compression_xy(image, x, y):
    # Ensure the image is in (C, H, W) format where C=4 (RGBA)
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"
    assert image.shape[1] > y, "Input height too large"
    assert image.shape[2] > x, "Input width too large"

    clipped_x = x - (x % 4)
    clipped_y = y - (y % 4)

    image_pad = image[:, clipped_y:clipped_y+4, clipped_x:clipped_x+4]
    print(image_pad)
    yuva_image = np.zeros_like(image_pad, dtype=int)
    for i in range(4):
        for j in range(4):
            r = image_pad[0, i, j]
            g = image_pad[1, i, j]
            b = image_pad[2, i, j]
            a = image_pad[3, i, j]
            t, u, v, a = rgba_to_yuva((r, g, b, a))
            yuva_image[0, i, j] = t
            yuva_image[1, i, j] = u 
            yuva_image[2, i, j] = v 
            yuva_image[3, i, j] = a 

    compressed_format = compress_pane(yuva_image)
    return min(compressed_format[0], 64) # returns size in bytes 

'''
    Returns a compressed image format which is just a list of blocks in row major order of the panes used by mali compression. 
    Each pane in the compressed format is represented by the (val_tree, bit count tree, and the pane size in the header)
'''
def mali_compression(image, height, width):

    # Ensure the image is in (C, H, W) format where C=4 (RGBA)
    assert image.shape[0] == 4, "Expected input image with 4 channels (RGBA)"

    # Ensure dimensions are multiples of 4
    # TODO should we increase to window size of 16 x 16 pixels 
    height = int(np.ceil(image.shape[1] / 4) * 4)
    width = int(np.ceil(image.shape[2] / 4) * 4)

    # Pad image to (4, height, width)
    image_pad = np.zeros((4, height, width), dtype=int)
    image_pad[:, :image.shape[1], :image.shape[2]] = image

    # Convert image to YUVA 
    # yuva_image = image_pad
    yuva_image = np.zeros_like(image_pad, dtype=int)
    for x in range(height):
        for y in range(width):
            r = image_pad[0, x, y]
            g = image_pad[1, x, y]
            b = image_pad[2, x, y]
            a = image_pad[3, x, y]
            t, u, v, a = rgba_to_yuva((r, g, b, a))
            yuva_image[0, x, y] = t
            yuva_image[1, x, y] = u 
            yuva_image[2, x, y] = v 
            yuva_image[3, x, y] = a 

    compressed_data = []

    # Compress pane by pane 
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            blk = yuva_image[:, y:y+4, x:x+4]  # Extract 4x4 block
            compressed_format = compress_pane(blk)
            # TODO need to translate the compressed format into a bitstream 
            compressed_data.append(compressed_format)

    return compressed_data

'''
    Takes a compressed form of the image using the mali compression and returns the original image 
'''
def mali_decompression(compressed_data, width, height):
    image_reconstructed = np.zeros((4, height, width), dtype=int)

    index = 0
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            compressed_blk = compressed_data[index]
            index += 1
            image_reconstructed[:, y:y+4, x:x+4] = decompress_pane(compressed_blk)

    # convert to RGBA 
    # rgba_image = image_reconstructed
    rgba_image = np.zeros_like(image_reconstructed, dtype=int)
    for x in range(height):
        for y in range(width):
            t = image_reconstructed[0, x, y]
            u = image_reconstructed[1, x, y]
            v = image_reconstructed[2, x, y]
            a = image_reconstructed[3, x, y]
            r, g, b, a = yuva_to_rgba((t, u, v, a))
            rgba_image[0, x, y] = r
            rgba_image[1, x, y] = g
            rgba_image[2, x, y] = b 
            rgba_image[3, x, y] = a 

    return rgba_image


if __name__ == "__main__":

    # Gradient + uniform image
    gradient = np.linspace(0, 64, num=4*4, dtype=np.uint8).reshape(4, 4)
    z = np.zeros((4, 4), dtype=np.uint8)
    alpha_channel = np.full((4, 4), 255, dtype=np.uint8)  # (4,4)
    # grayscale_rgba_panes = np.stack([gradient + 20, gradient + 23, z + 100, alpha_channel], axis=0)  # (4,4,4)
    # grayscale_rgba_panes = np.stack([gradient + 20, gradient + 23, gradient + 100, gradient+50], axis=0)  # (4,4,4)
    # grayscale_rgba_panes = np.stack([z+2, z+1, z + 100, alpha_channel], axis=0)  # (4,4,4)

    # Random image 
    grayscale_rgba_panes = np.zeros((4, 4, 4))
    for chan in range(4):
        for x in range(4):
            for y in range(4):
                grayscale_rgba_panes[chan, y, x] = random.randint(0, 255)


    print(grayscale_rgba_panes)
    print()

    # compressed_data = mali_compression(grayscale_rgba_panes, 4, 4) 
    compressed_data = mali_compression_xy(grayscale_rgba_panes, 1, 3)
    print(compressed_data) 
    print() 

    # rgba_image = mali_decompression(compressed_data, 4, 4) 

    # print(rgba_image)
    # print() 

    # assert((rgba_image == grayscale_rgba_panes).all())
