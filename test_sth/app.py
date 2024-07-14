import customtkinter as ctk
from matplotlib.figure import Figure
import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Function to be run in the background
def background_task():
    while True:
        print("Background task is running...")
        x_data.append(len(x_data))
        y_data.append(random.randint(0, 10))
        # time.sleep(1)

stop_event = threading.Event()
# Create a thread for the background task
background_thread = threading.Thread(target=background_task, args=(stop_event,))

# Set the thread as a daemon thread
background_thread.daemon = True

# Start the background thread
background_thread.start()



root = ctk.CTk()
root.title("Training App")
root.geometry("720x480")

root.minsize(720, 480)

frame1 = ctk.CTkFrame(root, width=300, height=200, corner_radius=10, fg_color="red")
frame2 = ctk.CTkFrame(root, width=300, height=200, corner_radius=10, fg_color="green")

frame1.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
frame2.grid(row=0, column=1, padx=1, pady=1, sticky="nsew")

nav_frame = ctk.CTkFrame(frame2, width=300, height=100, corner_radius=10, fg_color="pink")
nav_frame.pack(pady=2.5, padx=5, fill="x", expand=True)

content_frame = ctk.CTkFrame(frame2, width=300, height=500, corner_radius=5, fg_color="blue")
content_frame.pack(pady=2.5, padx=5, fill="both", expand=True)

data_button = ctk.CTkButton(frame1, text="Data", corner_radius=10,bg_color="transparent")
data_button.pack(fill="x", padx=1, pady=(100, 5))

model_button = ctk.CTkButton(frame1, text="Model", corner_radius=10,bg_color="transparent")
model_button.pack(fill="x", padx=1, pady=5)

# fig = Figure(figsize=(5, 4), dpi=100)
# ax = fig.add_subplot(111)
# ax.plot([1, 2, 3, 4, 5], [2, 3, 5, 7, 11])

# # Embed the figure in the frame
# canvas = FigureCanvasTkAgg(fig, master=content_frame)
# canvas.draw()
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data)

# Function to update the plot
def update(frame):
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()

    if len(x_data) >= 10:
        ani.event_source.stop()

    return line,


# Create the animation
ani = FuncAnimation(fig, update, blit=True, interval=10)

# Embed the figure in the frame
canvas = FigureCanvasTkAgg(fig, master=content_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

background_thread = threading.Thread(target=background_task)

# Set the thread as a daemon thread
background_thread.daemon = True

# Start the background thread
background_thread.start()

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()
stop_event.set()
background_thread.join()