
from .model import Model
from .batches import BatchGenerator

import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets the internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        self._prediction_bool = np.empty((0,), dtype=bool)

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray):
            raise ValueError

        target_len = target.shape[0]

        if prediction.shape[0] != target_len:
            raise ValueError

        prediction_bool = np.zeros(target.shape)

        for i in range(target_len):
            prediction_bool[i] = np.argmax(prediction[i]) == target[i]

        self._prediction_bool = np.concatenate([self._prediction_bool, prediction_bool], axis=0)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        return 'accuracy: {0:.3f}'.format(self.accuracy())
        # return something like "accuracy: 0.395"

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        accuracy = self.accuracy()

        if type(other) != type(self):
            raise TypeError

        return accuracy < other.accuracy()

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        accuracy = self.accuracy()

        if type(other) != type(self):
            raise TypeError

        return accuracy > other.accuracy()

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        data_len = self._prediction_bool.shape[0]

        return 0.0 if data_len == 0 else float(sum(self._prediction_bool) / data_len)