import json
import matplotlib.pyplot as plt
import numpy as np

output_file = 'data.json'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    array = np.array(dict[b'data'][1])
    # array = np.array(np.split(array, 3))
    # reshaped_array_1 = array[0].reshape((32,32))
    # reshaped_array_2 = array[1].reshape((32,32))
    # reshaped_array_3 = array[2].reshape((32,32))
    # img = np.array([reshaped_array_1,reshaped_array_2,reshaped_array_3])
    # img = img.transpose((1, 2, 0))
    reshaped_array = array.reshape((3, 32, 32))

# Transpose the array to get the desired shape (32, 32, 3)
    img = reshaped_array.transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(dict[b'labels'][1])
    plt.show()

    # return len(dict[b'data'][1])
    return dict.keys()


print(unpickle("..\\data\\cifar-10-batches-py\\test_batch"))
print(unpickle("..\\data\\cifar-10-batches-py\\batches.meta"))
print(unpickle("..\\data\\cifar-10-batches-py\\data_batch_4"))