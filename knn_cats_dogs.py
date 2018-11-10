if __name__ == '__main__':
    from .dlvc.datasets.pets import PetsDataset
    from .dlvc.dataset import Subset
    from .dlvc.ops import chain, vectorize, type_cast
    from .dlvc.batches import BatchGenerator
    from .dlvc.models.knn import KnnClassifier
    from .dlvc.test import Accuracy
    import numpy as np

    dataset_path = 'cifar-10-batches-py'
    training = PetsDataset(dataset_path, Subset.TRAINING)
    validation = PetsDataset(dataset_path, Subset.VALIDATION)
    test = PetsDataset(dataset_path, Subset.TEST)

    op = chain([
            vectorize(),
            type_cast(np.float32)
        ])

    training_bg = BatchGenerator(dataset=training, num=len(training), shuffle=False, op=op)
    validation_bg = BatchGenerator(dataset=validation, num=len(validation), shuffle=False, op=op)
    test_bg = BatchGenerator(dataset=test, num=len(test), shuffle=False, op=op)
    
    num_classes = training.num_classes()
    k_values = np.arange(1, 101, 15)

    for train_set in training_bg:

        best_accuracy = Accuracy()
        
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