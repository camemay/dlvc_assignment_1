
from ..model import Model

import numpy as np

class KnnClassifier(Model):
    '''
    k nearest neighbors classifier.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, k: int, input_dim: int, num_classes: int):
        '''
        Ctor.
        k is the number of nearest neighbors to consult (>= 1).
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        '''

        self._k = k
        self._input_dim = input_dim
        self._num_classes = num_classes

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return (0, self._input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self._num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        As training simply entails storing the data, the model is reset each time this method is called.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0 as there is no training loss to compute.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError
        
        if data.shape[1] != self._input_dim or labels.shape[0] != data.shape[0]:
            raise ValueError

        if any(labels > self._num_classes - 1):
            raise RuntimeError

        try:
            self._traindata = data
            self._trainlabels = labels
        except:
            raise RuntimeError

        return 0.0

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if not isinstance(data, np.ndarray):
            raise TypeError

        if data.shape[1] != self._input_dim:
            raise ValueError

        try:

            data_len = data.shape[0]
            train_data_len = self._traindata.shape[0]
            class_count = np.zeros((data_len, self._num_classes))

            for i in range(0, data_len):

                distance = np.zeros((train_data_len, 2))

                for j in range(0, train_data_len):
                    distance[j] = [np.sum(abs(data[i]-self._traindata[j])), self._trainlabels[j]]

                index_sort = np.argsort(distance[:,0], axis=0)
                distance = distance[index_sort]

                for k in range(0, self._k):
                    k_class = int(distance[k, 1])
                    class_count[i, k_class] = class_count[i, k_class] + 1

            return np.apply_along_axis(self._softmax, 1, class_count) 
        
        except:
            raise RuntimeError

    def _softmax(self, class_count):
        return np.exp(class_count) / sum(np.exp(class_count))






