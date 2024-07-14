from cifar_dataset_2 import *
from cifar_lib_0 import *

class CifarModel(Model):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.conv2d_1 = keras.layers.Conv2D(32,(3,3),activation='relu')
        self.batchnor_1 = keras.layers.BatchNormalization()
        self.mpool_1 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2d_2 = keras.layers.Conv2D(64,(3,3),activation='relu')
        self.batchnor_2 = keras.layers.BatchNormalization()
        self.mpool_2 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2d_3 = keras.layers.Conv2D(32,(3,3),activation='relu')
        self.batchnor_3 = keras.layers.BatchNormalization()
        self.mpool_3 = keras.layers.MaxPooling2D(pool_size=(2,2))

        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(16,activation='relu')

        self.output_layer = keras.layers.Dense(10,activation='softmax')

    def __call__(self, input):
        # Define the forward pass
        x = self.conv2d_1(input)
        x = self.batchnor_1(x)
        x = self.mpool_1(x)
        x = self.conv2d_2(x)
        x = self.batchnor_2(x)
        x = self.mpool_2(x)
        x = self.conv2d_3(x)
        x = self.batchnor_3(x)
        x = self.mpool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)

        return self.output_layer(x)
    
    def save(self, filepath, overwrite=True, zipped=True, **kwargs):
        model_dir = "./models"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        filepath = os.path.join(model_dir, filepath)

        return super().save(filepath, overwrite, zipped, **kwargs)

if __name__ == "__main__":
    model = CifarModel()

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    train_reader = CifarReader()
    val_reader = CifarReader()
    test_reader = CifarReader()

    train_reader('data_batch_1')
    train_reader('data_batch_2')
    train_reader('data_batch_3')
    train_reader('data_batch_4')

    val = val_reader('data_batch_5')

    test = test_reader('test_batch')

    train_dataset = CifarDataset(train_reader.get_dataset())
    print(train_dataset.__len__())

    val_dataset = CifarDataset(val_reader.get_dataset())
    print(val_dataset.__len__())

    test_dataset = CifarDataset(test_reader.get_dataset())
    print(test_dataset.__len__())

    model.fit(train_dataset.imgs, train_dataset.labels, epochs=2)

    model.save("model_1.keras")
    
