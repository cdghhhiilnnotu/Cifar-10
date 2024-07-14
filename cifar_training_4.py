from cifar_dataset_2 import *
from cifar_model_3 import *
from cifar_lib_0 import *

class CifarTraining():
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def training_step(self, model, imgs, labels):
        loss_fn = keras.losses.CategoricalCrossentropy()
        optimizer = keras.optimizers.Adam()

        with tf.GradientTape() as tape:
            predictions = model(np.expand_dims(imgs, axis=0))
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    def training(self, loss_arr, x_data):
        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for step in range(self.train_dataset.__len__()):
                # print(self.train_dataset.imgs[step].shape)
                # print(self.train_dataset.labels[step])
                label = self.train_dataset.labels[step]
                label = label.reshape((1, 10))
                loss = self.training_step(self.model, self.train_dataset.imgs[step], label)
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss.numpy()}")
                    loss_arr.append(loss.numpy())
                    x_data.append(len(loss_arr))

if __name__ == "__main__":
    model = CifarModel()

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

    val_dataset = CifarDataset(val_reader.get_dataset())

    test_dataset = CifarDataset(test_reader.get_dataset())

    training = CifarTraining(model, train_dataset, val_dataset, test_dataset)
    training.training()

