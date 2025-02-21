import numpy as np
from tqdm import tqdm

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

def rle_compress(arr):
    """Compresses a 2D NumPy array using Run-Length Encoding (RLE)."""
    flat_arr = arr.flatten()

    values = []
    counts = []

    prev_value = flat_arr[0]
    count = 1

    for val in flat_arr[1:]:
        if val == prev_value:
            count += 1
        else:
            values.append(prev_value)
            counts.append(count)
            prev_value = val
            count = 1

    values.append(prev_value)
    counts.append(count)

    return (np.array(values, dtype=arr.dtype), np.array(counts, dtype=np.int32)), arr.shape

def rle_decompress(compressed_data):
    """Decompresses a 2D NumPy array from RLE format."""
    (values, counts), shape = compressed_data
    flat_arr = np.repeat(values, counts)
    return flat_arr.reshape(shape)

dataset = np.random.rand(10**7, 3, 3)

identity = Aut(dataset, identity_compress, identity_decompress, name="identity")
rle = Aut(dataset, rle_compress, rle_decompress, name="rle")

# identity.formal_validate()
rle.formal_validate()
