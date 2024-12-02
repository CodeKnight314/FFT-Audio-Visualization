import numpy as np

## Copied from my FFT compression repository
## Cooley-Tukey FFT Implementation (O(n log n))
def FFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    if n % 2 > 0:
        raise ValueError("Array length must be a power of 2")
    even = FFT_1D_optimized(array[0::2])
    odd = FFT_1D_optimized(array[1::2])
    terms = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd])

## Cooley-Tukey Inverse FFT Implementation (O(n log n))
def IFFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    if n % 2 > 0:
        raise ValueError("Array length must be a power of 2")
    even = IFFT_1D_optimized(array[0::2])
    odd = IFFT_1D_optimized(array[1::2])
    terms = np.exp(2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd]) 

def padArray(array: np.array): 
    if len(array.shape) == 1: 
        new_len = 2 ** int(np.ceil(np.log2(array.shape[0])))
        padded_array = np.zeros((new_len), dtype=array.dtype)
        padded_array[:array.shape[0]] = array 
    elif len(array.shape) == 2: 
        if array.shape[0] == 1: 
            array = np.squeeze(array, axis=0)
        else: 
            array = np.squeeze(array, axis=1)
        new_len = 2 ** int(np.ceil(np.log2(array.shape[0])))
        padded_array = np.zeros((new_len), dtype=array.dtype)
        padded_array[:array.shape[0]] = array
    else: 
        padded_array = None 
        
    return padded_array

if __name__ == "__main__":
    test_array = np.random.rand(8)
    fft_result = FFT_1D_optimized(test_array)
    ifft_result = IFFT_1D_optimized(fft_result)
    print("Original array:", test_array)
    print("After FFT and IFFT:", ifft_result.real)