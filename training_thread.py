import math
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import random

from cifar_dataset_2 import *
from cifar_model_3 import *
from cifar_lib_0 import *
from cifar_training_4 import *

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
# training.training([])

loss = []
x_data = []

def background_task(stop_event, training):
    while not stop_event.is_set():
        print("Background task is running...")
        training.training(loss, x_data)
        print(loss)
    print("Background task has stopped.")

stop_event = threading.Event()

# Create a thread for the background task
background_thread = threading.Thread(target=background_task, args=(stop_event, training))

# Start the background thread
background_thread.start()

fig, ax = plt.subplots()
# x_data, y_data = [], []
line, = ax.plot(x_data, loss)

ax.ticklabel_format(useOffset=False)


# Function to update the plot
def update(frame):
    
    ax.set_ylim(0, 2 if len(loss) <= 0 else math.ceil(max(loss)))
    ax.set_xlim(0, 100)
    line.set_data(x_data, loss)
    ax.relim()
    ax.autoscale_view()

    # if len(x_data) > 10:
    #     ani.event_source.stop()

    return line,

# Create the animation
ani = FuncAnimation(fig, update, blit=True)
plt.show()

# Signal the background thread to stop
stop_event.set()

# Wait for the background thread to finish
background_thread.join()

print("Main program has finished.")


