if __name__ == '__main__':
    from dlvc.datasets.pets import PetsDataset
    from dlvc.dataset import Subset
    from dlvc.ops import chain, vectorize, type_cast, add, mul, hwc2chw, hflip, rcrop, blur, resize
    from dlvc.batches import BatchGenerator
    from dlvc.models.knn import KnnClassifier
    from dlvc.models.pytorch import CnnClassifier
    from dlvc.test import Accuracy
    #from dlvc.visualize import Plot as vplt
    import numpy as np
    import pdb
    import time
    import os

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    import cv2

   
    op_val = chain([
            resize(244),
            type_cast(np.float32),
            add(-127.5),
            mul(1/127.5),
            hwc2chw(),
        ])
    dataset_path = r'C:\Users\gallm\Documents\GitHub\cifar-10-batches-py'
    num_batches = 32
    in_shape=tuple((num_batches, 3, 32,32))
    test = PetsDataset(dataset_path, Subset.TEST)
    test_bg = BatchGenerator(dataset=test, num=len(test), shuffle=True, op=op_val)
    num_classes = test.num_classes()
    
    #net = torch.load(os.path.join(os.getcwd(), "best_model_densenet121_all_train.pth"))
    net = torch.load(os.path.join(os.getcwd(), "best_model_densenet121_all_train.pth"), map_location=lambda storage, loc: storage)


    final_clf = CnnClassifier(net=net,input_shape=in_shape, num_classes=num_classes, lr=0.01, wd=0.00001)
    accuracy2 = Accuracy()
    for test_set in test_bg:
        test_prediction = final_clf.predict(data=test_set.data)
        accuracy2.update(prediction=test_prediction, target=test_set.label)

    print("Test accuracy: {}".format(accuracy2.accuracy()))
