
from ..model import Model

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss and SGD are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (sgd with Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        # Setup cuda mode if cuda is available
        if torch.cuda.is_available():
             self._cuda = True
             self._cuda_device = torch.device(0)
             net.cuda()
        
        else:
            self._cuda = False
            self._cuda_device = None
        
        self._net = net
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._lr = lr
        self._wd = wd        

        self._loss = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._lr, momentum=0.9, weight_decay=self._wd, nesterov=True)        
            
        print("Cuda usage: {}".format(next(net.parameters()).is_cuda))  


    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return tuple(self._input_shape)


    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return tuple(self._num_classes,)


    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.   
        Raises RuntimeError on other errors.
        '''

        if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError

        if any(labels > self._num_classes - 1) or any(labels < 0):
            raise ValueError

        if data.shape[0] != labels.shape[0]:
            raise ValueError

        if data.dtype is not np.dtype(np.float32):
            raise ValueError

        try:
            self._net.train(True)

            if self._cuda:
                data = torch.tensor(data, device=self._cuda_device)
                labels = torch.tensor(labels, dtype=torch.int64, device=self._cuda_device)

            else:
                data = torch.tensor(data)
                labels = torch.tensor(labels, dtype=torch.int64)

            # Reset optimizer
            self._optimizer.zero_grad()

            out = self._net(data)
            loss_val = self._loss(out, labels)
            loss_val.backward()
            self._optimizer.step()

            return loss_val.item()
        except:
            raise RuntimeError

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if not isinstance(data, np.ndarray):
            raise TypeError

        if data.dtype is not np.dtype(np.float32):
            raise ValueError

        try:
            with torch.no_grad():

                # Transfer arrays into (cuda) tensors
                if self._cuda:
                    data = torch.tensor(data, device=self._cuda_device)           

                else:
                    data = torch.tensor(data) 

                self._net.eval()
                out = self._net(data)
                prob = F.softmax(out, dim=1)

                return prob.cpu().detach().numpy()
        except:
            raise RuntimeError