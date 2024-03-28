import datetime
import os
import threading
import tkinter as tk
from tkinter import *
import numpy as np
import pandas as pd
import ttkbootstrap as ttk
import cv2
from PIL import Image, ImageTk
import dlib

class AttendanceTracking:
    def __init__(self, root):

        # Window Initialize Variable
        self.root = root
        self.root.title("Attendance Tracking System")
        self.root.geometry("1080x500")

        # Frame Variable
        self.switch_frame = ttk.Frame(self.root)
        self.recognition_UI_frame = ttk.Frame(self.root)
        self.register_ui_frame = ttk.Frame(self.root)

        # General Variable
        self.name = None
        self.recognition_mode = False
        self.previous_time = datetime.datetime.now()
        # Face Recognition Variable
        
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10


        # Face Recognition Function
        self.video_capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # Init Function & Process
        self.get_face_database()
        self.wraper_switch_ui()
        self.draw_left_frame_camera()

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            return 1
        else:
            return 0
    
    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
    
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_switch_ui(self):
        # UI for switching between modes
        self.switch_frame.place(relx=0.75, rely=0.05, relwidth=0.2, relheight=0.25)
        self.switch_mode()
        face_recognition_btn = ttk.Checkbutton(self.switch_frame,
                                               text="Face Recognition",
                                               variable=BooleanVar(value=True),
                                               command=self.switch_mode,
                                               bootstyle="success-round-toggle")
        face_recognition_btn.pack(side="top", fill="both")

    def switch_mode(self):
        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            self.recognition_UI_frame = ttk.Frame(self.root)
            self.destroy_register_ui()
            self.draw_right_part_recognition_UI()
        else:
            self.register_ui_frame = ttk.Frame(self.root)
            self.destroy_face_recognition_ui()
            self.draw_right_part_register_info_ui()

    def update_image_recognition(self):
        if self.name:
            path = "./data/data_faces_from_camera"
            dir_list = os.listdir(path)
            for i in range(len(dir_list)):
                if f'person_{i}_{self.name}' in dir_list:
                    image_path = f'./data/data_faces_from_camera/person_{i}_{self.name}/img_face_2.jpg'
         # Load the original image
        original_img = Image.open(image_path)
        original_img = original_img.resize((200, 200), Image.ANTIALIAS)  # Resize the image if necessary

        # Load the frame image
        frame_img = Image.open("./images/frame.png")  # Replace "frame.png" with the path to your frame image
        frame_img = frame_img.resize((200, 200), Image.ANTIALIAS)  # Resize the frame image to match the original image size

        # Create a new image with the frame
        img_with_frame = Image.new("RGBA", (200, 200), (255, 255, 255, 0))  # Create a transparent image
        img_with_frame.paste(original_img, (0, 0))  # Paste the original image onto the transparent image
        img_with_frame.paste(frame_img, (0, 0), mask=frame_img)  # Paste the frame image onto the transparent image

        # Convert the new image to PhotoImage
        img_with_frame_tk = ImageTk.PhotoImage(img_with_frame)
        self.img_label.configure(image=img_with_frame_tk)
        self.img_label.image = img_with_frame_tk

    def draw_right_part_recognition_UI(self):
        self.recognition_UI_frame.place(relx=0.68, rely=0.2, relwidth=0.2, relheight=0.6)


        image_path = "./images/fake_profile.png"
        
        # Load the original image
        original_img = Image.open(image_path)
        original_img = original_img.resize((200, 200), Image.ANTIALIAS)  # Resize the image if necessary

        # Load the frame image
        frame_img = Image.open("./images/frame.png")  # Replace "frame.png" with the path to your frame image
        frame_img = frame_img.resize((200, 200), Image.ANTIALIAS)  # Resize the frame image to match the original image size

        # Create a new image with the frame
        img_with_frame = Image.new("RGBA", (200, 200), (255, 255, 255, 0))  # Create a transparent image
        img_with_frame.paste(original_img, (0, 0))  # Paste the original image onto the transparent image
        img_with_frame.paste(frame_img, (0, 0), mask=frame_img)  # Paste the frame image onto the transparent image

        # Convert the new image to PhotoImage
        img_with_frame_tk = ImageTk.PhotoImage(img_with_frame)

        # Create a label to display the image with frame
        self.img_label = tk.Label(self.recognition_UI_frame, image=img_with_frame_tk)
        self.save_fake_icons_tk = img_with_frame_tk
        self.img_label.image = img_with_frame_tk  # Keep a reference to prevent garbage collection
        self.img_label.pack(side='top', fill="both", expand=True)


        name = self.name
        self.user_name_btn = ttk.Label(self.recognition_UI_frame, text=f'UserName: {name}', bootstyle="success")
        self.user_name_btn.pack(fill="both", expand=True)

        submit_btn = ttk.Button(self.recognition_UI_frame,
                                text="Submit",
                                bootstyle = "success, outline",
                                command=self.prediction)
        submit_btn.pack(side='bottom', fill="both", expand=True)
        
    def update_display_name(self):
        self.user_name_btn.configure(text=f'UserName: {self.name}')


    def draw_right_part_register_info_ui(self):
        self.register_ui_frame.place(relx=0.57, rely=0.15, relwidth=1, relheight=1)
        header_register_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Face Register", 
                                        font=25, 
                                        bootstyle="success")
        step1_input_name_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Step 1: Input Name", 
                                        font=15, 
                                        bootstyle="success")
        input_name_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Name", 
                                        font=10, 
                                        bootstyle="success")
        
        input_name2_btn = ttk.Entry(self.register_ui_frame, 
                                        textvariable= "Input username",
                                        width=13,
                                        font=10, 
                                        bootstyle="success")
        
        step2_input_save_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Step 2: Save Face Image", 
                                        font=15, 
                                        bootstyle="success")
        save_current_face_btn = ttk.Button(self.register_ui_frame, 
                                        text = "Save Current Face", 
                                        bootstyle="success")
        
        header_register_btn.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        step1_input_name_btn.grid(row=1, column=0, padx=10, pady=10)
        input_name_btn.grid(row=2, column=0, pady=10)
        input_name2_btn.grid(row=2, column=1, pady=10)
        step2_input_save_btn.grid(row=3, column=0, padx=10, pady=10)
        save_current_face_btn.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
        
    def destroy_register_ui(self):
        self.register_ui_frame.destroy()
    
    def destroy_face_recognition_ui(self):
        self.recognition_UI_frame.destroy()

    
    
    def prediction(self, faces, img_rd):
        # 6.1  if cnt not changes
        if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                self.reclassify_interval_cnt != self.reclassify_interval):

            self.current_frame_face_position_list = []

            if "unknown" in self.current_frame_face_name_list:
                self.reclassify_interval_cnt += 1

            if self.current_frame_face_cnt != 0:
                for k, d in enumerate(faces):
                    self.current_frame_face_position_list.append(tuple(
                        [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                    self.current_frame_face_centroid_list.append(
                        [int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2])

                    img_rd = cv2.rectangle(img_rd,
                                            tuple([d.left(), d.top()]),
                                            tuple([d.right(), d.bottom()]),
                                            (255, 255, 255), 2)

            #  Multi-faces in current frame, use centroid-tracker to track
            if self.current_frame_face_cnt != 1:
                self.centroid_tracker()

        # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
        else:
            self.current_frame_face_position_list = []
            self.current_frame_face_X_e_distance_list = []
            self.current_frame_face_feature_list = []
            self.reclassify_interval_cnt = 0

            # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
            if self.current_frame_face_cnt == 0:
                # clear list of names and features
                self.current_frame_face_name_list = []
            # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
            else:
                self.current_frame_face_name_list = []
                for i in range(len(faces)):
                    shape = self.predictor(img_rd, faces[i])
                    self.current_frame_face_feature_list.append(
                        self.face_reco_model.compute_face_descriptor(img_rd, shape))
                    self.current_frame_face_name_list.append("unknown")

                # 6.2.2.1 Traversal all the faces in the database
                for k in range(len(faces)):
                    self.current_frame_face_centroid_list.append(
                        [int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2])

                    self.current_frame_face_X_e_distance_list = []

                    # 6.2.2.2  Positions of faces captured
                    self.current_frame_face_position_list.append(tuple(
                        [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                    # 6.2.2.3 
                    # print(self.face_features_known_list)
                    # For every faces detected, compare the faces in the database
                    for i in range(len(self.face_features_known_list)):
                        if str(self.face_features_known_list[i][0]) != '0.0':
                            e_distance_tmp = self.return_euclidean_distance(
                                self.current_frame_face_feature_list[k],
                                self.face_features_known_list[i])
                            self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                        else:
                            #  person_X
                            self.current_frame_face_X_e_distance_list.append(999999999)

                    # print(self.current_frame_face_X_e_distance_list)
                    # 6.2.2.4 / Find the one with minimum e distance
                    similar_person_num = self.current_frame_face_X_e_distance_list.index(
                        min(self.current_frame_face_X_e_distance_list))

                    if min(self.current_frame_face_X_e_distance_list) < 0.4:
                        self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                        
                        # Insert attendance record
                        nam =self.face_name_known_list[similar_person_num]
                        self.name = nam
                        self.update_image_recognition()
                        self.update_display_name()

    def draw_left_frame_camera(self):
        ret, frame = self.video_capture.read()
        img_rd = frame
        if ret:
            # Recognition
            faces = self.detector(frame, 0)
            
            for face in faces:
                cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            
            if self.recognition_mode:
                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                
                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                result_threading = threading.Thread(target=self.prediction, args=(faces, img_rd))
                result_threading.start()
                        
            # Display the updated frame
            frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_frame.config(image=frame)

            # Keep a reference to prevent garbage collection
            self.video_frame.image = frame  
        
        current_time = datetime.datetime.now()
        time_difference = current_time - self.previous_time
        threshold = datetime.timedelta(seconds=7)
        if self.current_frame_face_cnt > 0:
            self.previous_time = current_time
        if time_difference > threshold:
            self.name = None
            self.update_display_name()
            self.img_label.configure(image=self.save_fake_icons_tk)
            self.img_label.image = self.save_fake_icons_tk
            print("5 seconds have passed.")
            self.previous_time = current_time
            
        # Update every 10 milliseconds 
        self.root.after(10, self.update_video)  

    def wraper_switch_ui(self):
        # Draw the initial login UI

        # Draw the mode switch UI
        self.draw_switch_ui()  

        # Add other UI elements as needed
        self.video_frame = ttk.Label(self.root, borderwidth=5, relief='groove')
        self.video_frame.place(width=560, height=470, relx=0.03, rely=0.03)

    def update_video(self):
        self.draw_left_frame_camera()

    def main(self):
        self.root.mainloop()

root = ttk.Window(themename="superhero")
app = AttendanceTracking(root)
app.main()
