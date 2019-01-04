
import numpy as np
import random
import cv2

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return op

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.ravel(sample)

    return op

def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(sample, (2,0,1))
    
    return op

def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(sample, (1,2,0))
    
    return op

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample + val
    
    return op

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample * val
    
    return op

def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:    
        if random.randrange(0,100,1) < 50:    
            return np.flip(sample,1)
        else:
            return sample
    
    return op

def blur() -> Op:
    '''
    Blurs arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:   

        if random.randrange(0,100,1) < 50:    
            return cv2.GaussianBlur(sample, (3,3), cv2.BORDER_DEFAULT)

        else:
            return sample
    
    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    def op(sample: np.ndarray) -> np.ndarray:       

        if pad > 0:
            sample_pad = np.ndarray((sample.shape[0]+pad*2, sample.shape[1]+pad*2, sample.shape[2]))
            sample_pad[:,:,0] = np.pad(sample[:,:,0], pad_width=pad, mode=pad_mode)
            sample_pad[:,:,1] = np.pad(sample[:,:,1], pad_width=pad, mode=pad_mode)
            sample_pad[:,:,2] = np.pad(sample[:,:,2], pad_width=pad, mode=pad_mode)

            sample =  sample_pad.astype(np.uint8)

        if sz > sample.shape[0] or sz > sample.shape[1]:
            raise ValueError("Crop size of {} is bigger than padded image of shape {}.".format(sz, sample.shape))

        valid_start = (sample.shape[0]-sz, sample.shape[1]-sz)
        randx = np.random.randint(0,valid_start[0]+1)
        randy = np.random.randint(0,valid_start[1]+1)

        sample = sample[randx:randx+sz, randy:randy+sz,:] 

        return sample

    return op

def resize(sz: int) -> Op:
    '''
    Resize image to sz by zero padding
    '''

    def op(sample: np.ndarray) -> np.ndarray:       

        
        
        sample_pad = np.ndarray((sz, sz, sample.shape[2]))
        sample_pad[:,:,0] = cv2.resize(sample[:,:,0], dsize=(sz, sz), interpolation=cv2.INTER_CUBIC)
        sample_pad[:,:,1] = cv2.resize(sample[:,:,1], dsize=(sz, sz), interpolation=cv2.INTER_CUBIC)
        sample_pad[:,:,2] = cv2.resize(sample[:,:,2], dsize=(sz, sz), interpolation=cv2.INTER_CUBIC)

        sample =  sample_pad.astype(np.uint8)
      
        return sample

    return op
