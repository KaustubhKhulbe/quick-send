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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

IMAGE = "4.2.07.tiff"

cache_stats = {}

def sample(N, func, W, H):
    if func == "random":
        return [(random.randint(0, W-1), random.randint(0, H-1)) for _ in range(N)]
    
    elif func == "gradient":
        if N == 1:
            return [(0, 0)]  # or center if preferred
        return [
            (
                int(((i / (N - 1)) ** 2) * (W - 1)),
                int(((i / (N - 1)) ** 2) * (H - 1))
            )
            for i in range(N)
        ]
    elif func == "periodic":
        period = max(1, N // 10)
        return [( (i % period) * W // period, (i % period) * H // period ) for i in range(N)]
    
    elif func == "blurred":
        center_x, center_y = W // 2, H // 2
        return [(int(random.gauss(center_x, W//8)), int(random.gauss(center_y, H//8))) 
                for _ in range(N)]
    
    elif func == "skew":
        def clamp(val, limit):
            return max(0, min(val, limit - 1))
        
        return [
            (
                clamp(int((random.random() ** 2) * W), W),
                clamp(int((random.random() ** 0.5) * H), H)
            )
            for _ in range(N)
        ]
    
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
        mru_time = max(cache.values(), default=-1)
        cache[item] = mru_time + 1
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
        hits = 0
        misses = 0
        N=500
        points = sample(N, access_pattern, width, height)
        # Simulate 
        # print(len(points))
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
                hits += 1
                continue # send 0 bytes 
            else:
                misses += 1
                # total_bytes_sent += compression_algo(image[:, block_start_x:(block_start_x+self.block_size[0]), block_start_y:(block_start_y+self.block_size[1])], height, width)
                total_bytes_sent += compression_algo(image, x, y)
                cache = insert_into_cache(cache, cache_size, (block_start_x, block_start_y))

        # print(f"Total bytes sent: {total_bytes_sent}, Total bytes communicated: {total_bytes_communicated}")
        return (total_bytes_sent, total_bytes_communicated, total_bytes_communicated / total_bytes_sent, hits, misses)
 
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

image = tifffile.imread(f'dataset/{IMAGE}')
image = np.array(image, dtype=np.uint8)

# If grayscale (2D or single-channel), broadcast to RGB
if image.ndim == 2:
    # Shape (H, W) → (3, H, W)
    image = np.stack([image] * 3, axis=0)
    alpha = np.full((1, image.shape[1], image.shape[2]), 255, dtype=np.uint8)
    image = np.concatenate((image, alpha), axis=0)
else:
    image = tifffile.imread(f'dataset/{IMAGE}')
    image = np.transpose(np.array(image, dtype=np.uint8))
    image = np.concatenate((image, np.full((1, image.shape[1], image.shape[2]), 255, dtype=np.uint8)), axis=0)
    temp = Image.fromarray(np.moveaxis(image, 0, -1), mode="RGBA")

temp = Image.fromarray(np.moveaxis(image, 0, -1), mode="RGBA")
temp.save("test.png")

# print(image.shape)

dataset = np.array([image])
eval = Eval(dataset, block_size=(4, 8), compress_algos=algorithms["Intel_8gen"], access_patterns=patterns, cache_sizes=cache_sizes)

start_time = time.time()
results = {}
compressions = {}

# Helper function to calculate compressibility
def process_image(args):
    algo_name, algo_func, pattern, cache_size, idx, image = args
    sent, total, ratio, hits, misses = eval.calculate_compressability(image, algo_func, pattern, cache_size)
    return (algo_name, pattern, cache_size, idx, sent, total, ratio, hits, misses)

# Prepare all jobs ahead of time
jobs = []
for algo_name, algo_func in algorithms.items():
    for pattern in patterns:
        for cache_size in cache_sizes:
            for idx, image in enumerate(dataset):
                jobs.append((algo_name, algo_func, pattern, cache_size, idx, image))

# Run in parallel
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_image, job) for job in jobs]
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        algo_name, pattern, cache_size, idx, sent, total, ratio, hits, misses = future.result()

        key = (algo_name, pattern, cache_size)
        if key not in results:
            results[key] = [0, 0, 0, 0]
        results[key][0] += sent
        results[key][1] += total
        results[key][2] += hits
        results[key][3] += misses

        if (algo_name, idx) not in compressions:
            compressions[(algo_name, idx)] = 0
        compressions[(algo_name, idx)] += ratio

# Normalize compression values
scale = len(patterns) * len(cache_sizes)
for key in compressions:
    compressions[key] = compressions[key] / scale

end_time = time.time()

# Print Results
with open(f"results_{IMAGE}.csv", "w") as file:
    file.write("Algorithm,Pattern,Cache Size,Bytes Sent,Total Bytes,Compression Ratio,Hits,Misses\n")
    for key, value in results.items():
        file.write(f"{key[0]},{key[1]},{key[2]},{value[0]},{value[1]},{value[1] / value[0] if value[0] != 0 else 0},{value[2]},{value[3]}\n")

print(f"Total Time Taken: {end_time - start_time:.2f} seconds")