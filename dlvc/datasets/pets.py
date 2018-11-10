
from ..dataset import Sample, Subset, ClassificationDataset

import numpy as np
import pickle
import os

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    label_names = { 'cat': 0, 'dog': 1 }

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files

        if not os.path.isdir(fdir):
            raise ValueError

        pets_dict = self.__label_index(fdir)
        
        self.data = np.empty((0, 32, 32, 3))
        self.labels = np.empty((0), dtype=int)

        if (subset == Subset.TRAINING):
            vals = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4']
        elif (subset == Subset.VALIDATION):
            vals = ['data_batch_5']
        elif (subset == Subset.TEST):
            vals = ['test_batch']

        for val in vals:

            file_path = fdir + '/' + val

            if not os.path.isfile(file_path):
                raise ValueError

            with open(file_path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')

            images = np.transpose(np.reshape(dict[b'data'], (-1, 3, 32, 32)), (0,2,3,1))
            images = images[:, :, :, ::-1] # rgb to bgr
            labels = np.array(dict[b'labels'])

            selected_pets = [label in pets_dict.keys() for label in labels]
            labels = [self.label_names[pets_dict[label]] for label in labels[selected_pets]]

            self.labels = np.concatenate([self.labels, labels], axis=0)
            self.data = np.concatenate([self.data, images[selected_pets]], axis=0)

        self.labels = self.labels.astype('uint8')
        self.data = self.data.astype('uint8')
        
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        return Sample(idx, self.data[idx], self.labels[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return len(self.label_names.keys())

    def __label_index(self, fdir:str):

        with open(fdir + '/batches.meta', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        label_idx = {}

        for label in list(self.label_names.keys()):
            label_idx[label] = dict[b'label_names'].index(str.encode(label))

        label_idx = { y:x for x,y in label_idx.items()}

        return label_idx