import numpy as np
import math

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
            print(bits)
            if bits < 0: bits = 0

            print(header)
            header |= (0x07 & int(bits)) << (48 - 4 - 32 - 3*(i+1))
            print(header)
        
        header_string = f"{header:048b}"

        residual_data = np.zeros((32,), dtype=np.uint16)

        cl_str = ""
        i = 0
        for r in range(0, 4):
            for c in range(0, 8):
                bit_string = ""

                for channel in range(4):  # Iterate over channels
                    bit_width = int(compression_bits_required[channel])
                    # print(f"channel: {channel}, bit_width: {bit_width}")

                    if bit_width > 0:  # Skip channels with 0-bit storage
                        # print(blk.shape)
                        # print(mins.shape)

                        value = blk[channel][r][c] - mins[channel]  # Compute difference
                        bin_value = f"{value:0{bit_width}b}"  # Convert to binary

                        bit_string += bin_value  # Concatenate bits


                if len(bit_string) < 14: bit_string = "0" * (14 - len(bit_string)) + bit_string
                print(f"bit_string {r*8 + c}: {bit_string}")
                cl_str += bit_string

                if bit_string:  # Ensure bit_string isn't empty
                    residual_data[r * 8 + c] = int(bit_string, 2)  # Convert to integer
                else:
                    residual_data[r * 8 + c] = 0  # Default value if all channels are 0 bits


        lines = []
        flag = 0

        if (compressable):
            cache_lines = header_string
            # cache_lines = np.zeros((64, 1), dtype=np.uint8)
            # for i in range(6):
            #         cache_lines[i] = (header >> (8 * (5 - i))) & 0xFF

            bit_string = "".join(f"{num:014b}" for num in residual_data)  # 14-bit binary representation

            # for i in range(0, len(bit_string), 8):
            #     byte_chunk = bit_string[i:i+8]
            #     cache_lines[6 + i // 8] = int(byte_chunk, 2)

            cache_lines += cl_str + "0"*16
            lines = [cache_lines, cache_lines]
            flag = 1

        else:
            # cache_lines = blk.reshape(64, 2).astype(np.uint8)
            cache_lines = "0" * 48
            lines = [cache_lines, cache_lines]

        return lines, flag


def generate_test():
  random_blk = np.random.randint(4, 12, (4, 4, 8), dtype=np.uint8)
  with open("/home/kkhulbe2/Documents/cs534/quick-send/hardware/hvl/vcs/pixels.txt", "w") as f:
    # blk is of format (channel, row, col)
    # print pixels in (r, g, b, a) format
    for i in range(random_blk.shape[1]):
      for j in range(random_blk.shape[2]):
        f.write(f"{random_blk[0][i][j]} {random_blk[1][i][j]} {random_blk[2][i][j]} {random_blk[3][i][j]}\n")

  lines, flag = compress_4x8(random_blk)
  with open("/home/kkhulbe2/Documents/cs534/quick-send/hardware/hvl/vcs/expected_output.txt", "w") as f:
    for i in range(2):
      f.write(f"{lines[i]}\n")
    # make flag a 2-bit binary string
    f.write(f"0 {flag}\n")  

      

generate_test()


# 000000000100000001000000010000000100010010010010
# 000000000100000001000000010000000100010010010010