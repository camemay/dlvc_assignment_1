
import cv2
import numpy as np
import os
import time
from collections import namedtuple
import pdb 

Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self._fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self._fn = self._fn.astype(np.float32)
        self._fn /= (2**16-1)


    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image.
        Use the result to visualize the progress of gradient descent.
        '''

        vis = self._fn - self._fn.min()
        vis /= self._fn.max()
        vis *= 255
        vis = vis.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)

        return vis

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        loc1, loc2 = int(np.round(loc.x1)), int(np.round(loc.x2))

        if (loc1 >= self._fn.shape[0]) or (loc2 >= self._fn.shape[1]):
         raise ValueError()

        return self._fn[loc1, loc2]


def grad(fn: Fn, loc: Vec2, eps: float) -> Vec2:
    '''
    Compute the numerical gradient of a 2D function fn at location loc,
    using the given epsilon. See lecture 5 slides.
    Raises ValueError if loc is out of bounds of fn or if eps <= 0.
    '''
    
    if (loc.x1 >= fn._fn.shape[0]) or (loc.x2 >= fn._fn.shape[1]):
         raise ValueError()   

    if eps <= 0: raise ValueError  

    x1 = (fn(Vec2(loc.x1+eps, loc.x2))-fn(Vec2(loc.x1-eps, loc.x2)))/(2*eps)
    x2 = (fn(Vec2(loc.x1, loc.x2+eps))-fn(Vec2(loc.x1, loc.x2-eps)))/(2*eps)

    return Vec2(x1,x2)

def add_vec2(v1: Vec2, v2:Vec2):
    ''' Add two Vec2 '''
    return Vec2(v1.x1+v2.x1, v1.x2+v2.x2)

def sub_vec2(v1: Vec2, v2:Vec2):
    ''' Substract v2 from v1 (Vev2) '''
    return Vec2(v1.x1-v2.x1, v1.x2-v2.x2)

def prod_vec2(vec: Vec2, sk: float):
    ''' product of Vec2 with scalar '''
    return Vec2(vec.x1*sk, vec.x2*sk)

if __name__ == '__main__':
    # parse args

    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='SGD learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()
    
    # Initialize
    # loss fct
    fn = Fn(args.fpath)

    # visualisation
    vis = fn.visualize()

    # initial point
    loc = Vec2(args.sx1, args.sx2)

    velocity = Vec2(0, 0)

    while True:
        
        fgrad = grad(fn, loc, args.eps)

        # update direction
        if args.nesterov:
            velocity = sub_vec2(prod_vec2(velocity, args.beta), prod_vec2(fgrad, args.learning_rate))
            loc_next = add_vec2(loc, velocity)
        else:
            velocity = prod_vec2(fgrad, args.learning_rate)
            loc_next = sub_vec2(loc, velocity)

        # break if gradient is close to zero
        break_criteria = np.sqrt(sum(np.square([velocity.x1, velocity.x2])))
        if break_criteria < 0.01:
            break

        cv2.line(vis, (int(loc.x2), int(loc.x1)), (int(loc_next.x2), int(loc_next.x1)), color=(255,0,0), thickness=2)
        cv2.imshow('Progress', vis)
        cv2.waitKey(1)  # 20 fps, tune according to your liking

        loc = loc_next
