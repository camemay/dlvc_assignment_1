
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

    def __init__(self, net: nn.Module, input_shape: int, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (sgd with Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        # TODO implement

        # inside the train() and predict() functions you will need to know whether the network itself
        # runs on the cpu or on a gpu, and in the latter case transfer input/output tensors via cuda() and cpu().
        # do termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # you will want to initialize the optimizer and loss function here. note that pytorch's cross-entropy loss includes normalization so no softmax is required

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
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._lr, momentum=0.9, weight_decay=self._wd)        
            
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

        # TODO implement
        # make sure to set the network to train() mode
        # see above comments on cpu/gpu

        self._net.train()

        if self._cuda:
            data = torch.tensor(data, device=self._cuda_device)
            labels = torch.tensor(labels, dtype=torch.int64, device=self._cuda_device)

        else:
            data = torch.tensor(data)
            labels = torch.tensor(labels, dtype=torch.int64)

        self._optimizer.zero_grad()

        out = self._net(data)
        loss_val = self._loss(out, labels)
        loss_val.backward()
        self._optimizer.step()

        return loss_val.item()


    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement

        # pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # make sure to set the network to eval() mode
        # see above comments on cpu/gpu

        data = torch.tensor(data, device=self._cuda_device)     

        self._net.eval()
        out = self._net(data)
        prob = F.softmax(out, dim=1)

        return prob.cpu().detach().numpy()