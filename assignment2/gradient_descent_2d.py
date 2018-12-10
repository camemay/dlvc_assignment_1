
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
        print("____________________________")
        pdb.set_trace()
        loc1, loc2 = np.round(loc.x1), np.round(loc.x2)

        if (loc1 >= self._fn.shape[0]) or (loc2 >= self._fn.shape[1]):
         raise ValueError()
        print(loc1, loc2)
        return self._fn[int(loc1), int(loc2)]


def grad(fn: Fn, loc: Vec2, eps: float) -> Vec2:
    '''
    Compute the numerical gradient of a 2D function fn at location loc,
    using the given epsilon. See lecture 5 slides.
    Raises ValueError if loc is out of bounds of fn or if eps <= 0.
    '''
    
    # TODO implement one of the two versions presented in the lecture

    if (loc.x1 >= fn._fn.shape[0]) or (loc.x2 >= fn._fn.shape[1]):
         raise ValueError()

    if eps <= 0: raise ValueError    
    grad = Vec2(None,None)

    grad.x1 = (fn(Vec2(loc.x1+eps, loc.x2)-fn(Vec2(loc.x1-eps, loc.x2))))/(2*eps)
    print("-----------------------")
    grad.x2 = (fn(Vec2(loc.x1, loc.x2+eps)-fn(Vec2(loc.x1, loc.x2-eps))))/(2*eps)
    print("wwwwwwwwwwwwwwwwwwww-")
    return grad

def add_vec2(v1: Vec2, v2:Vec2):
    return Vec2(v1.x1+v2.x1, v1.x2+v2.x2)


def sub_vec2(v1: Vec2, v2:Vec2):
    return Vec2(v1.x1-v2.x1, v1.x2-v2.x2)

def prod_vec2(vec: Vec2, sk: float):
    return Vec2(vec.x1*sk, vec.x2*sk)



if __name__ == '__main__':
    # parse args

    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--step_size', type=float, default=10.0, help='Step size')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # init

    fn = Fn(args.fpath)
    vis = fn.visualize()
    loc = Vec2(int(args.sx1), int(args.sx2))
    grad(fn,Vec2(int(1),int(1)), args.eps)
    print("ENDE")
    pdb.set_trace()
    velo = Vec2(0,0)

    # perform gradient descent
    first = loc
    while True:
        # TODO implement normal gradient descent, with momentum, and with nesterov momentum depending on the arguments (see lecture 4 slides)
        # visualize each iteration by drawing on vis using e.g. cv2.line()
        # break out of loop once done

        # calc gradient from first @ theta+v
        fgrad = grad(fn, add_vec2(first,velo), args.eps)

        # get new v = beta*v - alpha*gradient(theta+v)
        velo = sub_vec2(prod_vec2(velo, args.nesterov), prod_vec2(args.step_size,fgrad))

        # new = old + v
        second = add_vec2(first,velo)

        cv2.line(vis, tuple(first), tuple(second), color=(255,0,0))
        cv2.imshow('Progress', vis)
        cv2.waitKey(50)  # 20 fps, tune according to your liking

        first = second