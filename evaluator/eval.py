import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pdb
import random

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

    '''
        Input: Takes an image, a compression algorithm, access_pattern, and a cache size 
        Output: tuple that is (sent bytes, total bytes commmunicated) 
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

        # Simulate 
        for access in access_pattern: 
            total_bytes_communicated = self.block_size[0] * self.block_size[1] * 4 # 4 bytes per pixel 

            x = access[0] 
            y = access[1]

            # Fetch that part of the image (convert coordinates to start of the compression block)
            block_start_x = x - (x % self.block_size[0]) 
            block_start_y = y - (y % self.block_size[1])

            # Determine if cached? 
            if(item_in_cache(cache, (block_start_x, block_start_y))):
                continue # send 0 bytes 
            else:
                total_bytes_sent += compression_algo(image[:, block_start_x:(block_start_x+self.block_size[0]), block_start_y:(block_start_y+self.block_size[1])])
                insert_into_cache(cache, cache_size, (block_start_x, block_start_y))

        return (total_bytes_sent, total_bytes_communicated) 
 
    # TODO: Go through all the possible configurations here

    # def formal_validate(self):
    #     print(f"Begining formal validation of AUT: {self.name}")
    #     for i in tqdm(range(len(self.images))):
    #         self.validate(self.images[i], self.images[i].shape[1], self.images[i].shape[2])
    #     print(f"AUT ({self.name}) formally validated")
