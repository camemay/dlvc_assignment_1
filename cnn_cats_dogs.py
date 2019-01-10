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

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 3 input image channels, 18 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)            
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            self.batch1 = nn.BatchNorm2d(16)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

            self.batch2 = nn.BatchNorm2d(32)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

            

            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(64*4*4, 128)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)

        def forward(self, x):
            #Computes the activation of the first convolution
            #Size changes from (3, 32, 32) to (16, 32, 32)

            x = F.relu(self.conv1(x))

            #Size changes from (16, 32, 32) to (16, 16, 16)
            x = self.pool(x)   

            #x = self.batch1(x)       

            #Size changes from (16, 16, 16) to (32, 16, 16)
            x = F.relu(self.conv2(x))

            #x = self.batch2(x) 

            #Size changes from (32, 16, 16) to (32, 8, 8)
            x = self.pool(x)  

            #Size changes from (32, 8, 8) to (64, 8, 8)
            x = F.relu(self.conv3(x))

            #Size changes from (64, 8, 8) to (64, 4, 4)
            x = self.pool(x)          

            #Reshape data to input to the input layer of the neural net
            #Size changes from (32, 8, 8) to (1, 2048)
            #Recall that the -1 infers this dimension from the other given dimension
            x = x.view(-1, 64 * 4 * 4)

            #Computes the activation of the first fully connected layer
            #Size changes from (1, 2084) to (1, 64)
            x = F.relu(self.fc1(x))

            #x = self.dropout(x)           

            #Computes the activation of the first fully connected layer
            #Size changes from (1, 64) to (1, 2)
            x = F.relu(self.fc2(x))

            x = self.fc3(x)

            return x


    dataset_path = 'cifar-10-batches-py'
    training = PetsDataset(dataset_path, Subset.TRAINING)
    validation = PetsDataset(dataset_path, Subset.VALIDATION)
    test = PetsDataset(dataset_path, Subset.TEST)
    
    op = chain([
            #blur(),
            hflip(),
            rcrop(32,4,"constant"),
            resize(244),
            type_cast(np.float32),
            add(-127.5),
            mul(1/127.5),
            hwc2chw(),
        ])

    op_val = chain([
            resize(244),
            type_cast(np.float32),
            add(-127.5),
            mul(1/127.5),
            hwc2chw(),
        ])
    
    num_batches = 32
    in_shape=tuple((num_batches, 3, 32,32))

    training_bg = BatchGenerator(dataset=training, num=num_batches, shuffle=True, op=op)
    validation_bg = BatchGenerator(dataset=validation, num=num_batches, shuffle=True, op=op_val)
    test_bg = BatchGenerator(dataset=test, num=len(test), shuffle=True, op=op_val)
    
    num_classes = training.num_classes()

    '''
    transfer learning
    '''
    net = torchvision.models.densenet121(pretrained=True)

    # #Freeze parameters so we don't backprop through them
    # for param in net.parameters():
    #     param.requires_grad = False
        
    # net.classifier = nn.Sequential(nn.Linear(1024, 256),
    #                                 nn.ReLU(),
    #                                 nn.Dropout(0.2),
    #                                 nn.Linear(256, 2))
    net = net.cuda()

    # if not transfer learning:
    #net = Net()   

    clf = CnnClassifier(net=net, input_shape=in_shape, num_classes=num_classes, lr=0.001, wd=0.0001)

    acc_best = [-1, -1]

    #plot = vplt("Model A")
    #plot.register_scatterplot("Loss", "Epoch", "Loss")
    totstart = time.time()
    for epoch in range(1,101):
        
        epstart = time.time()        
        losses = []

        for train_set in training_bg:

            loss = clf.train(data=train_set.data, labels=train_set.label)            
            losses.append(loss)
        
        accuracy = Accuracy()

        for val_set in validation_bg:
            val_prediction = clf.predict(data=val_set.data)
            accuracy.update(prediction=val_prediction, target=val_set.label)

        if acc_best[0] < accuracy.accuracy():
            torch.save(net, os.path.join(os.getcwd(), "best_model.pth"))
            acc_best[0] = accuracy.accuracy()
            acc_best[1] = epoch

        epstop = time.time()
        losses = np.asarray(losses)
        loss_mean= np.mean(losses)

        #plot.update_scatterplot("Loss", epoch, loss)
        print("epoch {} ({:.3} s)".format(epoch, epstop-epstart))
        print("   train loss: {:.3f} +- {:.3f}".format(loss_mean, np.std(losses)))
        print("   val acc:    {:.3f}".format(accuracy.accuracy()))    
    
    totstop = time.time()
    print("Best Accuracy of {} at epoch {}".format(acc_best[0], acc_best[1]))
    print("Training duration: {:.3} min".format((totstop-totstart)/60))
    print("Applying test-set...")

    final_clf = CnnClassifier(net=torch.load(os.path.join(os.getcwd(), "best_model.pth")),input_shape=in_shape, num_classes=num_classes, lr=0.01, wd=0.00001)
    accuracy2 = Accuracy()
    for test_set in test_bg:
        test_prediction = final_clf.predict(data=test_set.data)
        accuracy2.update(prediction=test_prediction, target=test_set.label)

    print("Test accuracy: {}".format(accuracy2.accuracy()))
