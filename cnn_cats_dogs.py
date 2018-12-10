if __name__ == '__main__':
    from dlvc.datasets.pets import PetsDataset
    from dlvc.dataset import Subset
    from dlvc.ops import chain, vectorize, type_cast, add, mul, hwc2chw
    from dlvc.batches import BatchGenerator
    from dlvc.models.knn import KnnClassifier
    from dlvc.models.pytorch import CnnClassifier
    from dlvc.test import Accuracy
    import numpy as np
    import pdb

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 3 input image channels, 18 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(64*16*16, 64)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            #Computes the activation of the first convolution
            #Size changes from (3, 32, 32) to (18, 32, 32)
            x = F.relu(self.conv1(x))

            #Size changes from (18, 32, 32) to (64, 32, 32)
            x = F.relu(self.conv2(x))

            #Size changes from (64, 32, 32) to (64, 16, 16)
            x = self.pool(x)

            #Reshape data to input to the input layer of the neural net
            #Size changes from (64, 16, 16) to (1, 16384)
            #Recall that the -1 infers this dimension from the other given dimension
            x = x.view(-1, 64 * 16 *16)

            #Computes the activation of the first fully connected layer
            #Size changes from (1, 16384) to (1, 64)
            x = F.relu(self.fc1(x))

            #Computes the activation of the first fully connected layer
            #Size changes from (1, 64) to (1, 2)
            x = F.relu(self.fc2(x))

            return x


    dataset_path = r'C:\Users\gallm\Documents\GitHub\cifar-10-batches-py'
    training = PetsDataset(dataset_path, Subset.TRAINING)
    validation = PetsDataset(dataset_path, Subset.VALIDATION)
    #test = PetsDataset(dataset_path, Subset.TEST)
    
    op = chain([
            type_cast(np.float32),
            add(-127.5),
            mul(1/127.5),
            hwc2chw(),
        ])

    num_batches = 128
    in_shape=tuple((num_batches, 3, 32,32))

    training_bg = BatchGenerator(dataset=training, num=num_batches, shuffle=False, op=op)
    validation_bg = BatchGenerator(dataset=validation, num=num_batches, shuffle=False, op=op)
    #test_bg = BatchGenerator(dataset=test, num=len(test), shuffle=False, op=op)
    
    num_classes = training.num_classes()
    net = Net()
    # model = model.cuda()

    clf = CnnClassifier(net=net, input_shape=in_shape, num_classes=num_classes, lr=0.05, wd=0.1)
    
    for epoch in range(0,101):
        
        for train_set in training_bg:

            # There is an ERRRROR in Train!!!! 
            #loss = clf.train(data=train_set.data, labels=train_set.label)
            
            '''
            accuracy = Accuracy()
            
            for k in k_values:
                print('Training k = {0}'.format(k))
                kNN = KnnClassifier(k=k, input_dim=train_set.data[0].shape[0], num_classes=num_classes)
                kNN.train(data=train_set.data, labels=train_set.label)

                accuracy = Accuracy()

                for val_set in validation_bg:
                    val_prediction = kNN.predict(data=val_set.data)
                    accuracy.update(prediction=val_prediction, target=val_set.label)
                    
                print(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_k = k
            
            print('best k = {0} with VALIDATION {1}'.format(best_k, best_accuracy))

            kNN = KnnClassifier(k=best_k, input_dim=train_set.data[0].shape[0], num_classes=num_classes)
            kNN.train(data=train_set.data, labels=train_set.label)

            accuracy.reset()

            for test_set in test_bg:
                test_prediction = kNN.predict(data=test_set.data)
                accuracy.update(prediction=test_prediction, target=test_set.label)

            print('best k = {0} with TESTING {1}'.format(best_k, accuracy))
            '''