import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import random

x_data = []
y_data = []
# Function to be run in the background
def background_task(stop_event):
    while not stop_event.is_set():
        print("Background task is running...")
        x_data.append(len(x_data))
        y_data.append(random.randint(0, 10))
        time.sleep(1)
    print("Background task has stopped.")

# Create an Event object to signal the thread to stop
stop_event = threading.Event()

# Create a thread for the background task
background_thread = threading.Thread(target=background_task, args=(stop_event,))

# Start the background thread
background_thread.start()

fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data)

# Function to update the plot
def update(frame):
    
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()

    if len(x_data) > 10:
        ani.event_source.stop()

    return line,

# Create the animation
ani = FuncAnimation(fig, update, blit=True)
plt.show()

# Signal the background thread to stop
stop_event.set()

# Wait for the background thread to finish
background_thread.join()

print("Main program has finished.")