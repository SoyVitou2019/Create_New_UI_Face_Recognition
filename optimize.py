from tkinter import *
import tkinter as tk
import ttkbootstrap as ttk
from PIL import Image, ImageTk
import threading
import dlib
import cv2

class AttendanceTracking(tk.Tk):
    def __init__(self):
        super().__init__()
        # Initialize Window
        self.title(" IHUB Attendance Application ")
        self.geometry("900x500")
        self.icon_path = "./images/ihub.png"
        self.iconphoto(True, tk.PhotoImage(file=self.icon_path))

        # Init function
        self.boolean_init()
        self.camera_init()
        self.main_frame()
        self.main_widget()
        self.main_place_frame()
        self.main_process()

    def boolean_init(self):
        self.recognition_mode = True

    def camera_init(self):
        self.video_capture = cv2.VideoCapture(0)

    def main_frame(self):
        self.camera_frame = ttk.Frame(self)

    def main_place_frame(self):
        self.camera_frame.place(relx=0.01, rely=0.01, relwidth=1, relheight=1)
        self.video_frame.place(width=600, height=470, relx=0.01, rely=0.03)

    def main_widget(self):
        self.video_frame = ttk.Label(self.camera_frame, relief='groove')

    def main_destroy_widget(self):
        pass

    def main_destroy_frame(self):
        pass

    def main_process(self):
        ret, frame = self.video_capture.read()
        img_rd = frame
        if ret:
            # Display the updated frame
            frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_frame.config(image=frame)

        # Update every 10 milliseconds 
        self.after(10, self.main_process)  # <-- Corrected here

# Create a Tk instance for the main window
if __name__ == "__main__":
    app = AttendanceTracking()  # <-- Corrected here
    app.mainloop()  # <-- Corrected here
