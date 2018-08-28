from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure
import Detection.k_means as k_means

# Set up window, loading widgets, option widgets
window = Tk()
greyscale = IntVar()
figure = Figure()

window.title("Detection GUI")
window.geometry('500x100')

load_frame = Frame(window)
load_frame.pack()

file_box = Entry(load_frame)
file_box.pack(side=LEFT)


def load_file():
    file_box.delete(0, END)
    file_box.insert(0, filedialog.askopenfilename())


load_button = Button(load_frame, text="Load", width=10, command=load_file)
load_button.pack(side=LEFT)

option_frame = Frame(window)
option_frame.pack()

cluster_label = Label(option_frame, text="Number of clusters: ")
cluster_label.pack(side=LEFT)
cluster_box = Spinbox(option_frame, from_=2, to=100)
cluster_box.pack(side=LEFT)

greyscale_cb = Checkbutton(option_frame, variable=greyscale)
greyscale_cb.pack(side=RIGHT)
greyscale_label = Label(option_frame, text="Split image into 2-color layers?")
greyscale_label.pack(side=RIGHT)


def run():
    if greyscale:
        k_means.get_greyscale_layers(file_box.get(), int(cluster_box.get()))
    else:
        k_means.get_clustered_image(file_box.get(), int(cluster_box.get()))


run_button = Button(window, text="Run", width=10, command=run)
run_button.pack()


window.mainloop()


