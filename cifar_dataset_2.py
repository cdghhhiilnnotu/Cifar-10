from typing import Any
from cifar_lib_0 import *

class CifarReader():
    def __init__(self, datapath='.\\data\\cifar-10-batches-py'):
        self.labels = []
        self.imgs = []
        self.datapath = datapath

    def __call__(self, filepath):
        with open(os.path.join(self.datapath, filepath), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        
        img_arr = np.array([img for img in dict[b'data']])

        for array in img_arr:
            reshaped_array = array.reshape((3, 32, 32))
            img = reshaped_array.transpose((1, 2, 0))
            self.imgs.append(img)
        
        for lbl in dict[b'labels']:
            label = np.zeros(10)
            label[int(lbl)] = 1.0
            self.labels.append(label)
        
        # plt.imshow(self.imgs[1])
        # plt.title(self.labels[1])
        # plt.show()

    def get_dataset(self):
        return self.imgs, self.labels
    
class CifarDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.imgs = np.array(dataset[0])/255.
        self.labels = np.array(dataset[1])/1.

    def __len__(self):
        return self.imgs.shape[0]


if __name__ == "__main__":
    # CifarDataset()
    train_reader = CifarReader()
    val_reader = CifarReader()
    test_reader = CifarReader()

    train_reader('data_batch_1')
    train_reader('data_batch_2')
    train_reader('data_batch_3')
    train_reader('data_batch_4')
    # train = train_reader.get_dataset()
    # print(np.array(train[0]).shape)
    # print(np.array(train[1]).shape)

    val = val_reader('data_batch_5')
    # val = val_reader.get_dataset()
    # print(np.array(val[0]).shape)
    # print(np.array(val[1]).shape)

    test = test_reader('test_batch')
    # test = test_reader.get_dataset()
    # print(np.array(test[0]).shape)
    # print(np.array(test[1]).shape)


    train_dataset = CifarDataset(train_reader.get_dataset())
    print(train_dataset.__len__())

    val_dataset = CifarDataset(val_reader.get_dataset())
    print(val_dataset.__len__())

    test_dataset = CifarDataset(test_reader.get_dataset())
    print(test_dataset.__len__())

    