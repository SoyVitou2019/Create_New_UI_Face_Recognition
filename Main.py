import datetime
import os
import shutil
import threading
import tkinter as tk
from tkinter import *
import numpy as np
import pandas as pd
import ttkbootstrap as ttk
import cv2
from PIL import Image, ImageTk
import dlib
from features_extraction_to_csv import feature_extraction
from mysql_query import MysqlQuery

class AttendanceTracking:
    def __init__(self, root):

        # Window Initialize Variable
        self.root = root
        self.root.title("Attendance Tracking System")
        self.root.geometry("1080x500")

        # Mysql configure
        self.mysql_connection = MysqlQuery(host="47.129.17.36", user="vitou_wct", password="vitou2357.!", database="attendance_test1")

        # Frame Variable
        self.switch_frame = ttk.Frame(self.root)
        self.recognition_UI_frame = ttk.Frame(self.root)
        self.register_ui_frame = ttk.Frame(self.root)

        # General Variable
        self.name = None
        self.recognition_mode = False
        self.previous_time = datetime.datetime.now()
        self.progress_bar = None
        self.register_face_processing = False
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

        # Register Variable
        self.current_face_dir = ""
        self.existing_faces_cnt = 0 # cnt for counting saved faces
        self.input_name_char = ""
        self.current_frame_faces_cnt = 0  #  cnt for counting faces in current frame
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.out_of_range_flag = False
        self.face_folder_created_flag = False
        self.ss_cnt = 0   #  cnt for screen shots
        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0
        # GUI part
        self.frame_right_info = tk.Frame(self.root)
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        

        # Face Recognition Function
        self.video_capture = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # Init Function & Process
        self.get_face_database()
        self.wraper_switch_ui()
        self.process()
    
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            self.face_name_known_list = []
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
        self.switch_frame.place(relx=0.60, rely=0.05, relwidth=1, relheight=0.4)
        self.switch_mode()

        manage_database_btn = ttk.Button(self.switch_frame,
                                         text="Database Management",
                                         bootstyle = "success, outline",
                                         command=self.database_management_toplevel)
        face_recognition_btn = ttk.Checkbutton(self.switch_frame,
                                               text="Face Recognition",
                                               variable=BooleanVar(value=True),
                                               command=self.switch_mode,
                                               bootstyle="success-round-toggle")
        manage_database_btn.grid(row=0, column=1, padx=10, pady=10)
        face_recognition_btn.grid(row=0, column=0, padx=10, pady=10)

    def database_management_toplevel(self):
        folder_path = self.path_photos_from_camera
        # Database Management UI variables
        self.management_toplevel = tk.Toplevel()
        self.management_toplevel.title("Database management")
        self.management_toplevel.geometry("700x500")
        self.selected_folder = tk.StringVar()
        self.selected_folder.set(folder_path)
        self.file_listbox = tk.Listbox(self.management_toplevel,
                                       selectmode=tk.MULTIPLE)

        # list down users
        self.file_listbox.pack(fill=tk.BOTH,padx=20, expand=True)

        self.populate_folders(folder_path)
        delete_folder_btn = ttk.Button(self.management_toplevel, 
                                       text="Delete Users",
                                       command=self.delete_folder)
        delete_folder_btn.pack(side='bottom', padx=10, pady=20)

    def populate_folders(self, folder_path):
        for item in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, item)):
                # Extract just the folder name
                folder_name = os.path.basename(item)
                self.file_listbox.insert(ttk.END, folder_name)

    def delete_folder(self):
        selected_items = self.file_listbox.curselection()
        selected_folders = [self.file_listbox.get(index) for index in selected_items]
        users_deleted = "User Deleted: \n"
        for index, x in enumerate(selected_folders):
            users_deleted += f"Index {index + 1}. {x}, \n"

        for folder_name in selected_folders:
            # Your code to delete the folder goes here
            folder_path = os.path.join(self.path_photos_from_camera, folder_name)
            try:
                # Recursive deletion of folder and its content
                shutil.rmtree(folder_path)
                self.register_feature_fn()
                print("Folder deleted:", folder_path)
                # Remove UI elements after deletion
                self.file_listbox.delete(0, tk.END)
                self.management_toplevel.destroy()
                # Show a pop-up modal upon successful deletion
                
                self.show_popup("Users Deleted", f" {users_deleted} \n Users deleted successfully!")

            except OSError as e:
                print(f"Error: {folder_path} : {e.strerror}")

    def show_popup(self, title, message):
        popup = tk.Toplevel()
        popup.title(title)
        popup.geometry("400x300")
        label = ttk.Label(popup, text=message, bootstyle="success")
        label.place(relx=0.5, rely=0.3, anchor="center")

        ok_button = ttk.Button(popup, text="<--- OK --->", bootstyle="success, outline", command=popup.destroy)
        ok_button.place(relx=0.5, rely=0.7, anchor="center")
        
    def switch_mode(self):
        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            self.get_face_database()
            self.recognition_UI_frame = ttk.Frame(self.root)
            self.destroy_register_ui()
            self.draw_right_part_recognition_UI()
        else:
            self.register_ui_frame = ttk.Frame(self.root)
            self.destroy_face_recognition_ui()
            self.draw_right_part_register_info_ui()

    def update_image_recognition(self):
        if self.img_label.winfo_exists():
            image_path = "./images/fake_profile.png"
            if self.name:
                dir_list = os.listdir(self.path_photos_from_camera)
                if self.name in dir_list:
                    image_path = f'./data/data_faces_from_camera/{self.name}/img_face_1.jpg'
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
        self.user_name_btn = ttk.Label(self.recognition_UI_frame,
                                        text=f'Username: {name}',
                                        font=25,
                                        bootstyle="warning")
        self.user_name_btn.pack(fill="both", expand=True)

        submit_btn = ttk.Button(self.recognition_UI_frame,
                                text="Submit attendance",
                                bootstyle = "success, outline",
                                command=self.submit_data_into_database)
        submit_btn.pack(side='bottom', fill="both", expand=True)
        
    def update_display_name(self):
        if self.user_name_btn.winfo_exists():
            self.user_name_btn.configure(text=f'Username: {self.name}')

    def submit_data_into_database(self):
        self.mysql_connection.write_data_into_attendance(self.name)
        style = "success"
        message = "Successfully"
        if self.name == None:
            style = "danger"
            message = "Fail to submit attendance"
        popup = tk.Toplevel()
        popup.title("Attendance")
        popup.geometry("500x300")
        label = ttk.Label(popup, text=message, bootstyle=style, font=("Helvetica", 20))
        label.place(relx=0.5, rely=0.3, anchor="center")
        self.name = None
        self.update_display_name()
        self.img_label.configure(image=self.save_fake_icons_tk)
        self.img_label.image = self.save_fake_icons_tk
        

        ok_button = ttk.Button(popup, text="<--- OK --->", bootstyle="success, outline", command=popup.destroy)
        ok_button.place(relx=0.5, rely=0.7, anchor="center")

    def draw_right_part_register_info_ui(self):
        self.register_ui_frame.place(relx=0.57, rely=0.15, relwidth=1, relheight=1)
        header_register_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Face Register", 
                                        font=45, 
                                        bootstyle="success")
        step1_input_name_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Step 1: Input Name", 
                                        font=15, 
                                        bootstyle="warning")
        input_name_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Full-Name: ", 
                                        font=10, 
                                        bootstyle="success")
        
        self.input_name2_btn = ttk.Entry(self.register_ui_frame, 
                                        textvariable= "Input username",
                                        width=13,
                                        font=10, 
                                        bootstyle="success")
        
        step2_input_save_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Step 2: Save Face Image", 
                                        font=15, 
                                        bootstyle="warning")
        save_current_face_btn = ttk.Button(self.register_ui_frame, 
                                        text = "Save Current Face",
                                        command=self.save_current_face,
                                        bootstyle="success, outline")
        step3_input_save_btn = ttk.Label(self.register_ui_frame, 
                                        text = "Step 3: Compile Image", 
                                        font=15, 
                                        bootstyle="warning")
        
        compile_feature_btn = ttk.Button(self.register_ui_frame, 
                                        text = "Register to Database",
                                        command=self.register_feature_fn,
                                        bootstyle="success, outline")
        
        header_register_btn.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        step1_input_name_btn.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        input_name_btn.grid(row=2, column=0, pady=10)
        self.input_name2_btn.grid(row=2, column=1, pady=10)
        step2_input_save_btn.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        save_current_face_btn.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
        step3_input_save_btn.grid(row=5, column=0, padx=10, pady=10, sticky='w')
        compile_feature_btn.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

    def draw_progress_bar(self):
        self.progress_bar = ttk.Progressbar(self.register_ui_frame, 
                                            mode="determinate", 
                                            maximum=200)
        self.progress_bar.grid(row=6, column=0, columnspan=3, padx=10, pady=10)
        self.progress_bar.start(10)


        
    def register_feature_fn(self):
        self.feature_thread = threading.Thread(target=feature_extraction)
        self.feature_thread.start()
        self.register_face_processing = True
        self.during_feature_thread_process()

    def during_feature_thread_process(self):
        popup = tk.Toplevel(self.root)
        popup.title("Processing")
        popup.geometry("500x300")
        
        message = "Loading..."
        style = "success"
        label = ttk.Label(popup, text=message, bootstyle=style, font=("Helvetica", 20))
        label.place(relx=0.5, rely=0.3, anchor="center")

        self.check_thread(popup, label)
       

    def destroy_register_ui(self):
        self.register_ui_frame.destroy()
    
    def destroy_face_recognition_ui(self):
        self.recognition_UI_frame.destroy()

    def get_name_from_entry(self):
        input_value = self.input_name2_btn.get()
        self.input_name_char = input_value.replace(" ", '-').lower()
    
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

                    # Important Threshold prediction confident > 60%
                    if min(self.current_frame_face_X_e_distance_list) < 0.4:
                        self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                        
                        # Insert attendance record
                        nam =self.face_name_known_list[similar_person_num]
                        self.name = nam


    # Face Register Part

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.get_name_from_entry()
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera
        os.makedirs(self.current_face_dir)

        self.ss_cnt = 0  #  Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created
        
    def save_current_face(self):
        self.get_name_from_entry()
        filenames = os.listdir(self.path_photos_from_camera)
        already_created = False
        if self.input_name_char in filenames:
            already_created = True
        if not already_created:
            self.create_face_folder()
        
        message = "Fail to save your face"
        style = "danger"
  
        if self.current_frame_faces_cnt == 1:
            if not self.out_of_range_flag:
                self.ss_cnt += 1
                #  Create blank image according to the size of face detected
                self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                np.uint8)
                for ii in range(self.face_ROI_height * 2):
                    for jj in range(self.face_ROI_width * 2):
                        self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                            self.face_ROI_width_start - self.ww + jj]
                self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)

                cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                message = "Successfully"
                style = "success"
                
            else:
                self.log_all["text"] = "Please do not out of range!"
        else:
            self.log_all["text"] = "No face in current frame!"

        popup = tk.Toplevel()
        popup.title("Save face modal")
        popup.geometry("500x300")
        label = ttk.Label(popup, text=message, bootstyle=style, font=("Helvetica", 20))
        label.place(relx=0.5, rely=0.3, anchor="center")


    def process(self):
        ret, frame = self.video_capture.read()
        img_rd = frame
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            # Recognition
            faces = self.detector(frame, 0)
            
            for face in faces:
                cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            if self.recognition_mode:
                current_time = datetime.datetime.now()
                time_difference = current_time - self.previous_time
                threshold = datetime.timedelta(seconds=10)
                if self.current_frame_face_cnt > 0:
                    self.previous_time = current_time
                if time_difference > threshold:
                    self.name = None
                    self.update_display_name()
                    self.img_label.configure(image=self.save_fake_icons_tk)
                    self.img_label.image = self.save_fake_icons_tk
                    print("5 seconds have passed.")
                    self.previous_time = current_time

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                
                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                if len(faces) < 2:
                    result_threading = threading.Thread(target=self.prediction, args=(faces, img_rd))
                    result_threading.start()
                    self.update_image_recognition()
                    self.update_display_name()
                if len(faces) > 2:
                    print("user > 2")
            else:
                self.label_face_cnt["text"] = str(len(faces))
                
                #  Face detected
                if len(faces) != 0 and len(faces) < 2:
                    #   Show the ROI of faces
                    for k, d in enumerate(faces):
                        self.face_ROI_width_start = d.left()
                        self.face_ROI_height_start = d.top()
                        #  Compute the size of rectangle box
                        self.face_ROI_height = (d.bottom() - d.top())
                        self.face_ROI_width = (d.right() - d.left())
                        self.hh = int(self.face_ROI_height / 2)
                        self.ww = int(self.face_ROI_width / 2)

                        # If the size of ROI > 480x640
                        if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                            self.label_warning["text"] = "OUT OF RANGE"
                            self.label_warning['fg'] = 'red'
                            self.out_of_range_flag = True
                            color_rectangle = (255, 0, 0)
                        else:
                            self.out_of_range_flag = False
                            self.label_warning["text"] = ""
                            color_rectangle = (255, 255, 255)
                        self.current_frame = cv2.rectangle(self.current_frame,
                                                        tuple([d.left() - self.ww, d.top() - self.hh]),
                                                        tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                        color_rectangle, 2)
                #   Show the ROI of faces
                self.current_frame_faces_cnt = len(faces)
                        
            # Display the updated frame
            frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_frame.config(image=frame)

            # Keep a reference to prevent garbage collection
            self.video_frame.image = frame  
        
        
            
        # Update every 10 milliseconds 
        self.root.after(10, self.update_video)  
        

    def check_thread(self, popup, label):
        if self.feature_thread.is_alive():
            self.root.after(100, self.check_thread, popup, label)
        else:
            self.register_face_processing = False
            label.config(text="Feature extraction completed", bootstyle="info")
            # You can also add a button to close the popup after completion
            close_button = ttk.Button(popup, text="Close", command=popup.destroy)
            close_button.place(relx=0.5, rely=0.7, anchor="center")
            # Call the method to get the face database after the thread is done
            self.get_face_database()

    def wraper_switch_ui(self):
        # Draw the initial login UI

        # Draw the mode switch UI
        self.draw_switch_ui()  

        # Add other UI elements as needed
        self.video_frame = ttk.Label(self.root, borderwidth=5, relief='groove')
        self.video_frame.place(width=600, height=470, relx=0.01, rely=0.03)

    def update_video(self):
        self.process()

    def main(self):
        self.root.mainloop()



root = ttk.Window(themename="darkly")
app = AttendanceTracking(root)
app.main()
