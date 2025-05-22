import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry('600x100')
root.title('Slider Demo')

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)

l_tresh = tk.IntVar()
u_tresh = tk.IntVar()

def get_current_value_1():
    return l_tresh.get()

def get_current_value_2():
    return u_tresh.get()

def slider_changed(event):
    value_label_1.configure(text=get_current_value_1())
    value_label_2.configure(text=get_current_value_2())
    print("l_tresh",l_tresh.get())
    print("u_tresh",u_tresh.get())


def print_value(val):
    print(f"Slider Value: {val}")

slider_label_1 = ttk.Label(root, text='Lower Thresh:')
slider_label_1.grid(column=0, row=0, sticky='w')

slider_1 = ttk.Scale(root, from_=0, to=255, orient='horizontal', variable=l_tresh, command=slider_changed)
slider_1.grid(column=1, row=0, sticky='we')

slider_label_2 = ttk.Label(root, text='Upper Thresh:')
slider_label_2.grid(column=0, row=1, sticky='w')

slider_2 = ttk.Scale(root, from_=0, to=255, orient='horizontal', variable=u_tresh, command=slider_changed)
slider_2.grid(column=1, row=1, sticky='we')

current_value_label = ttk.Label(root, text='Current Values:')
current_value_label.grid(row=2, columnspan=2, sticky='n', ipadx=10, ipady=10)

value_label_1 = ttk.Label(root, text=get_current_value_1())
value_label_1.grid(row=3, columnspan=2, sticky='n')

value_label_2 = ttk.Label(root, text=get_current_value_2())
value_label_2.grid(row=4, columnspan=2, sticky='n')


root.mainloop()
