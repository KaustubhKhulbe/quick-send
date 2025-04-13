from PIL import Image
import numpy as np
import tifffile
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pdb
import random
from mali import mali_compression
from igpu_11gen import igpu_11gen
from igpu_8gen import igpu_8gen

def sample(N, func, W, H):
    if func == "random":
        return [(random.randint(0, W-1), random.randint(0, H-1)) for _ in range(N)]
    
    elif func == "gradient":
        return [(int((i / N) * (W-1)), int((i / N) * (H-1))) for i in range(N)]
    
    elif func == "periodic":
        period = max(1, N // 10)
        return [( (i % period) * W // period, (i % period) * H // period ) for i in range(N)]
    
    elif func == "blurred":
        center_x, center_y = W // 2, H // 2
        return [(int(random.gauss(center_x, W//8)), int(random.gauss(center_y, H//8))) 
                for _ in range(N)]
    
    elif func == "skew":
        return [(int((random.random() ** 2) * (W-1)), int((random.random() ** 0.5) * (H-1))) 
                for _ in range(N)]
    
    else:
        raise ValueError(f"Unknown pattern: {func}")

'''
    Inserts item into cache, may evict items from the cache
'''
def insert_into_cache(cache, cache_size, new_item):
    cache_list = list(cache.items()) 

    # Grab MRU time 
    mru_time = -1 
    if(len(cache_list) != 0):        
        mru_time = max(cache_list, key=lambda x: x[1])[1]

    # insert into cache
    if(len(cache_list) < cache_size): # Just insert if we are less than cache size
        cache_list.append((new_item, mru_time + 1)) 
    else: # evict LRU entry 
        evict_idx = min(enumerate(cache_list, 0), key=lambda x: x[1][1])[0] 
        cache_list[evict_idx] = (new_item, mru_time + 1) 

    return dict(cache_list)

'''
    Check if item exists in the cache
'''
def item_in_cache(cache, item): 
    if(item in cache):
        return True
    else:
        return False
    
class Eval:
    '''
    images: numpy array where first dimension is an index to the image
    compress_algos: compression algorithm: f(compression block) -> compressed block size 
    compression_block_size: tuple of the pixel block width and height 
    access_patterns: List of image indexes that are being accessed
    cache_sizes: Size of the cache on the GPU (unit is the pixel block size)
    '''
    def __init__(self, images, compress_algos, block_size, access_patterns, cache_sizes):
        self.images = images 
        self.compress_algos = compress_algos 
        self.access_patterns = access_patterns
        self.cache_sizes = cache_sizes
        self.block_size = block_size # (width, height)

    '''
        Input: Takes an image, a compression algorithm, access_pattern, and a cache size 
        Output: tuple that is (sent bytes, total bytes commmunicated) 
        access_pattern: ["random", "gradient", "uniform", "periodic", "blurred", "skew"]
    ''' 
    def calculate_compressability(self, image, compression_algo, access_pattern, cache_size): 
        assert(image.shape[0] == 4) # There should be four channels
        width = image.shape[1]
        height = image.shape[2] 

        # Output 
        total_bytes_sent = 0 
        total_bytes_communicated = 0 

        # Cache (use LRU)
        cache = {}
        N=1000
        points = sample(N, access_pattern, width, height)
        # Simulate 
        print(len(points))
        for access in points: 
            total_bytes_communicated += self.block_size[0] * self.block_size[1] * 4 # 4 bytes per pixel 

            x = access[0] 
            y = access[1]

            if (x >= width or y >= height):
                print(f"Access out of bounds: {x}, {y}")
                continue
            elif (x < 0 or y < 0):
                print(f"Access out of bounds: {x}, {y}")
                continue

            # Fetch that part of the image (convert coordinates to start of the compression block)
            block_start_x = x - (x % self.block_size[0]) 
            block_start_y = y - (y % self.block_size[1])

            # Determine if cached? 
            if(item_in_cache(cache, (block_start_x, block_start_y))):
                continue # send 0 bytes 
            else:
                # total_bytes_sent += compression_algo(image[:, block_start_x:(block_start_x+self.block_size[0]), block_start_y:(block_start_y+self.block_size[1])], height, width)
                total_bytes_sent += compression_algo(image, x, y)
                insert_into_cache(cache, cache_size, (block_start_x, block_start_y))

        print(f"Total bytes sent: {total_bytes_sent}, Total bytes communicated: {total_bytes_communicated}")
        return (total_bytes_sent, total_bytes_communicated, total_bytes_communicated / total_bytes_sent)
 
    # TODO: Go through all the possible configurations here

    # def formal_validate(self):
    #     print(f"Begining formal validation of AUT: {self.name}")
    #     for i in tqdm(range(len(self.images))):
    #         self.validate(self.images[i], self.images[i].shape[1], self.images[i].shape[2])
    #     print(f"AUT ({self.name}) formally validated")


# Configurations
algorithms = {
    # 'Uncompressed': uncompressed,   # Yet to be defined
    'Intel_8gen': igpu_8gen.get_compressability_igpu_8gen,
    # 'Intel_11gen': igpu_11gen.compress_image,
    # 'Mali': mali_compression.mali_compression
}

patterns = ["random", "gradient", "periodic", "blurred", "skew"]
cache_sizes = [4, 8, 16, 32]

# dataset = np.random.randint(0, 3, (1, 4, 4, 8), dtype=np.uint8)
image = tifffile.imread('dataset/4.2.07.tiff')
image = np.transpose(np.array(image, dtype=np.uint8))
image = np.concatenate((image, np.full((1, image.shape[1], image.shape[2]), 255, dtype=np.uint8)), axis=0)
temp = Image.fromarray(np.moveaxis(image, 0, -1), mode="RGBA")
temp.save("test.png")

# print(image.shape)

dataset = np.array([image])
eval = Eval(dataset, block_size=(4, 8), compress_algos=algorithms["Intel_8gen"], access_patterns=patterns, cache_sizes=cache_sizes)

results = {}
compressions = {}

for algo_name, algo_func in algorithms.items():
    for pattern in patterns:
        for cache_size in cache_sizes:
            # for image in tqdm(dataset, desc=f"{algo_name} - {pattern} - Cache {cache_size}"):
            for idx, image in enumerate(tqdm(dataset, desc=f"{algo_name} - {pattern} - Cache {cache_size}")):
                sent, total, ratio = eval.calculate_compressability(image, algo_func, pattern, cache_size)
                results[(algo_name, pattern, cache_size)] = results.get((algo_name, pattern, cache_size), [0, 0])
                results[(algo_name, pattern, cache_size)][0] += sent
                results[(algo_name, pattern, cache_size)][1] += total
                # compressions[(algo_name, idx)] += ratio
                if idx not in compressions:
                    compressions[(algo_name, idx)] = ratio
                compressions[(algo_name, idx)] += ratio
            
for key, value in compressions.items():
    compressions[key] = compressions[key] / len(algorithms) / len(patterns) / len(cache_sizes)
            

# Print Results
for key, value in results.items():
    print(f"Algorithm: {key[0]}, Pattern: {key[1]}, Cache Size: {key[2]} => Bytes Sent: {value[0]}, Total Communicated: {value[1]}")
    print(f"Compression Ratio: {value[1] / value[0] if value[0] != 0 else 0}")