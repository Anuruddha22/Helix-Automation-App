# helix_controller.py
import cv2
import numpy as np
import logging
from ids_peak import ids_peak as peak
from ids_peak import ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl
import time
import SPiiPlusPython as sp
import csv
import socket
from ctypes import *
from datetime import datetime
import pandas as pd
from tifffile import imwrite
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from efficientnet_pytorch import EfficientNet
import pyvisa
from pyvisa import constants
import vxi11
from vxi11.vxi11 import Vxi11Exception
import re
from openpyxl import load_workbook
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import clr
from pathlib import Path
from PIL import Image

#----------------------------------------------------
# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
_logger = logging.getLogger(__name__)
#----------------------------------------------------


from motion_stage import MotionStage
from delay_generator import Delay_Generator
from pdv import PDV
from top_camera import Top_Camera, run_camera_workflow, disconnect_camera
from cnn_model import EfficientNetSegmentation
from power_meter import PowerMeter
from constants import SAMPLE_HOME_X, SAMPLE_HOME_Y, MM_PER_PIXEL, FLYER_RADIUS_PIXEL, PULSE_TIME, MODEL_IMAGE_SIZE
from waveplate_rotator import WaveplateRotator


class HelixController:
    def __init__(self):
        self.exp_type: str = ""
        self.flyer_id: str = ""
        self.flyer_material: str = ""
        self.flyer_thickness: str = ""
        self.sample_id: str = ""
        self.sample_material: str = ""
        self.spacing: str = "600"
        self.current_notes: str = ""

        # Now user inputs
        self.current_waveplate_angle: float = 0.0
        self.waveplate_angle_list: float = 0.0
        self.pdv_file_number: str = ""
        self.beam_file_number: str = ""

        self.previous_waveplate_angle: float = 0.0
        self.final_coordinates: list = []

        # Initialize constants
        self.home_x = SAMPLE_HOME_X
        self.home_y = SAMPLE_HOME_Y
        self.mmPpix = MM_PER_PIXEL
        self.flyer_radius = FLYER_RADIUS_PIXEL
        self.pulse_time = PULSE_TIME
        self.model_image_size = MODEL_IMAGE_SIZE

        # Initialize the hardware components
        self.top_camera = Top_Camera()
        self.delay_generator = Delay_Generator()
        self.motion_stage = MotionStage()
        self.power_meter = PowerMeter()
        self.pdv = PDV()
        self.waveplate_rotator = WaveplateRotator() 

        # initialize CNN model
        self.cnn_model = EfficientNetSegmentation()
        # Load the saved model weights from the specified path
        self.cnn_model.load_state_dict(torch.load('support_files/cnn_model.pth', map_location=torch.device('cpu')))
        # Set the model to evaluation mode, which is crucial when making predictions
        # This disables dropout and batch normalization behaviors that are different during training
        self.cnn_model.eval()
        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the input image (tensor or array) to a PIL image for easier resizing
            transforms.Resize((224, 224)),  # Resize the image to 224x224, the input size expected by EfficientNet
            transforms.ToTensor(),  # Convert the image back to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet pre-trained values
        ])

    def load_parameters(self, parameters: dict):
        self.exp_type = parameters.get("exp_type", "")
        self.flyer_id = parameters.get("flyer_id", "")
        self.sample_id = parameters.get("sample_id", "")
        self.flyer_material = parameters.get("flyer_material", "")
        self.sample_material = parameters.get("sample_material", "")
        self.flyer_thickness = parameters.get("flyer_thickness", "")
        self.spacing = parameters.get("spacing", "")
        self.current_notes = parameters.get("notes", "")
        self.pdv_file_number = parameters.get("pdv_file_number", "")
        self.beam_file_number = parameters.get("beam_file_number", "")
        angles_str = parameters.get("waveplate_angle", "30")
        try:
            self.waveplate_angle_list = [float(a.strip()) for a in angles_str.split(",")]
        except Exception as e:
            print("Invalid waveplate angle input, defaulting to 30 deg for all.")
            self.waveplate_angle_list = [30.0]

    def connect(self):
        """Connect to all hardware components."""
        try:
            run_camera_workflow(self.top_camera)
            self.delay_generator.connect()
            self.motion_stage.connect(self.motion_stage)
            self.power_meter.connect()
            self.pdv.connect()
            self.waveplate_rotator.connect()
            _logger.info("All components connected successfully.")
        except Exception as e:
            _logger.error(f"Error connecting components: {e}")

    def disconnect(self):
        self.motion_stage.disconnect(self.motion_stage)
        # self.pdv.disconnect()
        # disconnect_camera()
        self.delay_generator.disconnect()
        self.power_meter.disconnect()
        self.waveplate_rotator.disconnect()
        _logger.info("All components disconnected successfully.")

    #-------------------- Utility functions for generating filenames and logging experiments
    def get_next_pdv_filename(self, existing_df, flyer_id, date_str="20250711", prefix="C1"):
        # Filter matching entries for flyer and date
        pattern = f"{prefix}--{date_str}--(\\d+)"
        filtered = existing_df[existing_df["Flyer_ID"] == flyer_id]
        matches = filtered["PDV_FileName"].dropna().astype(str).str.extract(pattern)
        matches = matches.dropna()[0].astype(int) if not matches.empty else []

        next_num = matches.max() + 1 if not matches.empty else int(self.pdv_file_number)
        return f"{prefix}--{date_str}--{next_num:05d}"

    def get_next_beamprofile_filename(self, existing_df, flyer_id, date_str="20250711", prefix="Beam"):
        # Filter matching entries for flyer and date
        pattern = f"{prefix}--{date_str}--(\\d+)"
        filtered = existing_df[existing_df["Flyer_ID"] == flyer_id]
        matches = filtered["Beam_Profile_FileName"].dropna().astype(str).str.extract(pattern)
        matches = matches.dropna()[0].astype(int) if not matches.empty else []

        next_num = matches.max() + 1 if not matches.empty else int(self.beam_file_number)
        return f"{prefix}--{date_str}--{next_num:05d}"


    def log_experiment_to_excel(self, exp_type, flyer_id, flyer_material, flyer_thickness, sample_id, sample_material, spacing, waveplate_angle, pdv_rl1, pdv_rp1, pdv_rl2, pdv_rp2, pdv_tp, fr, fc, goal_x, goal_y, c_goal_x, c_goal_y, laser_energy, total_energy, shot_time,
                            notes="", flag=1):

        date_str = datetime.now().strftime("%Y%m%d")
        EXCEL_PATH = f"C:/Users/Administrator/Documents/HELIX/SeeMoveShootData/experiment_log_{exp_type}_{date_str}_{flyer_material}_{flyer_id}.xlsx"

        if os.path.exists(EXCEL_PATH) and flag==1:
            df_existing = pd.read_excel(EXCEL_PATH)
            pdv_file_name = self.get_next_pdv_filename(df_existing, flyer_id, date_str)
            beam_file_name = self.get_next_beamprofile_filename(df_existing, flyer_id, date_str)
            next_id = df_existing["Exp_ID"].max() + 1
        elif os.path.exists(EXCEL_PATH) and flag==0:
            df_existing = pd.read_excel(EXCEL_PATH)
            pdv_file_name = []
            beam_file_name = []
            next_id = df_existing["Exp_ID"].max() + 1
        elif flag==0:
            df_existing = pd.DataFrame()
            pdv_file_name = []
            beam_file_name = []
            next_id = 1
        else:
            df_existing = pd.DataFrame()
            pdv_file_name = f"C1--{date_str}--{self.pdv_file_number}"
            beam_file_name = f"Beam--{date_str}--{self.beam_file_number}"
            next_id = 1
        
        data = {
            "Timestamp": datetime.now(),
            "Exp_ID": next_id,
            "Flyer_ID": flyer_id,
            "Flyer_material": flyer_material,
            "Flyer_Thickness (um)": flyer_thickness,
            "Sample_ID": sample_id,
            "Sample material": sample_material,
            "Spacing (um)": spacing,
            "Waveplate_Angle (Degrees)": waveplate_angle,
            "PDV_FileName": pdv_file_name,
            "PDV_Target_Wavelength (m)": pdv_rl1,
            "PDV_Target_Power (dBm)": pdv_rp1,
            "PDV_Ref_Wavelength (m)": pdv_rl2,
            "PDV_Ref_Power (dBm)": pdv_rp2,
            "PDV_Return_Power (dBm)": pdv_tp,
            "Flyer_Row": fr,
            "Flyer_Column": fc,
            "Flyer_X_Position_Desired (mm)": goal_x,
            "Flyer_Y_Position_Desired (mm)": goal_y,
            "Flyer_X_Position_Corrected (mm)": c_goal_x,
            "Flyer_Y_Position_Corrected (mm)": c_goal_y,
            "Laser_Ref_Energy (mJ)": laser_energy,
            "Laser_Target_Energy (mJ)": total_energy,
            "Beam_Profile_FileName": beam_file_name,
            "Shot_Time (seconds)": shot_time,
            "Notes": notes
        }

        df_new = pd.DataFrame([data])
        if df_existing.empty:
            df_new.to_excel(EXCEL_PATH, index=False)
        else:
            with pd.ExcelWriter(EXCEL_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df_new.to_excel(writer, index=False, header=False, startrow=len(df_existing)+1)
        _logger.info(f"Experiment logged to {EXCEL_PATH} with ID {next_id}")


    def move_initial(self, x, y):
        # Blind estimated motion to desired flyer position 
        # Calculate the goal positions for x and y based on the input circle coordinates
        # Write condition for moving to circles
        # home_y -d : up, home_y +d: down, home_x -d: left, home_x + d: right 

        goal_x = self.home_x - 6 * x
        goal_y = self.home_y + 6 * (6 - y)

        goal_x = round(goal_x,4)
        goal_y = round(goal_y,4)
        self.motion_stage.move_to_point(goal_x, goal_y)
        return goal_x, goal_y 
    

    def detect_flyer(self):
        # Capture the current image frame
        frame = self.top_camera.get_image()

        if frame is None:
            raise ValueError("Top camera returned None.")

        # Ensure 3-channel image
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Invalid image shape: {frame.shape}")

        frame = frame[..., :3]  # Ensure it's a 3-channel image

        try:
            with torch.no_grad():
                frame0 = frame.copy()
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                new_image = self.transform(frame2)
                new_image = new_image.unsqueeze(0)

                # Predict mask and bounding box using the model
                mask, box = self.cnn_model(new_image)

                # Safety check for shape
                if mask is None or box is None:
                    raise ValueError("Model output is None")

                mask = mask.squeeze().cpu().numpy()
                box = box.squeeze().cpu().numpy()

                if len(box) != 4:
                    raise ValueError(f"Bounding box shape invalid: {box}")

        except Exception as e:
            print("Error during CNN prediction or processing:", e)
            raise

        # Convert bounding box values to pixel coordinates
        x1 = int((box[0] / self.model_image_size) * 960)
        y1 = int((box[1] / self.model_image_size) * 960)
        x2 = int(((box[0] + box[2]) / self.model_image_size) * 960)  
        y2 = int(((box[1] + box[3]) / self.model_image_size) * 960)  

        x0 = int((x1 + x2) / 2)
        y0 = int((y1 + y2) / 2)
        center = (x0, y0)

        # Draw the bounding box on the image
        cv2.rectangle(frame0, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame0, center

    def align_flyer(self, goal_x, goal_y, center):
        tx = (480 - center[0]) * self.mmPpix
        ty = (480 - center[1]) * self.mmPpix
        c_goal_x = goal_x + tx
        c_goal_y = goal_y + ty
        c_goal_x = round(c_goal_x, 4)
        c_goal_y = round(c_goal_y, 4)
        self.motion_stage.move_to_point(c_goal_x, c_goal_y)
        # time.sleep(1)
        return c_goal_x, c_goal_y
    
    def shoot_or_skip_flyer_with_pdv_reading(self, x, y, goal_x, goal_y, c_goal_x, c_goal_y, move_start, waveplate_angle):
        pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = self.pdv.read_PDV()
        pdv_tar_p = float(pdv_tar_p)
        if -20 < pdv_tar_p < 0:
            ##########################
            self.delay_generator.pulse_on() # Pulse on
            read_start = time.time()
            timeout = 4  # seconds
            initial_energy_val = self.power_meter.read_energy()
            current_energy_val = initial_energy_val
            while time.time() - read_start < timeout:
                time.sleep(0.1)  
                new_energy_val = self.power_meter.read_energy()
                if new_energy_val != current_energy_val:
                    current_energy_val = new_energy_val
                    break  
            self.delay_generator.pulse_off() # Pulse off
            energymJ = current_energy_val * 1e3
            energymJ = round(energymJ, 2)
            energyT = 11.13 * energymJ + 31.59
            energyT = round(energyT, 2)
            print(f"Measured Reference Energy (mJ): {energymJ}")
            print(f"Measured Total Energy (mJ): {energyT}")
            #############################

            shot_time = time.time() - move_start
            shot_time = round(shot_time, 2)
            print(f'Total time for shot: {shot_time}')
            self.laser_triggered = True  # Set flag to prevent re-triggering

            # frame_a = self.top_camera.get_image()
            # cv2.imwrite(f'Images/frame_{x}_{y}_j.jpg', frame_a)

            current_notes = 'Sucessful laser shot'
            print('Sucessful laser shot')
            self.log_experiment_to_excel(
                    exp_type = self.exp_type,
                    flyer_id=self.flyer_id, 
                    flyer_material=self.flyer_material, 
                    flyer_thickness=self.flyer_thickness, 
                    sample_id=self.sample_id, 
                    sample_material=self.sample_material, 
                    spacing=self.spacing, 
                    waveplate_angle=waveplate_angle,
                    pdv_rl1=pdv_ref_w1,
                    pdv_rp1=pdv_ref_p1,
                    pdv_rl2=pdv_ref_w2,
                    pdv_rp2=pdv_ref_p2,
                    pdv_tp=pdv_tar_p,
                    fr=x,
                    fc=y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    c_goal_x=c_goal_x,
                    c_goal_y=c_goal_y,
                    laser_energy=energymJ,
                    total_energy = energyT,
                    shot_time=shot_time,
                    notes=current_notes,
                    flag=1)
        else:
            print('Skipping due to bad PDV return')
            shot_time = time.time() - move_start
            shot_time = round(shot_time, 2)
            current_notes = 'Skipping due to bad PDV return'
            self.laser_triggered = False
            energymJ = 0
            energyT = 0
            self.log_experiment_to_excel(
                    exp_type = self.exp_type,
                    flyer_id=self.flyer_id, 
                    flyer_material=self.flyer_material, 
                    flyer_thickness=self.flyer_thickness, 
                    sample_id=self.sample_id, 
                    sample_material=self.sample_material, 
                    spacing=self.spacing, 
                    waveplate_angle=waveplate_angle,
                    pdv_rl1=pdv_ref_w1,
                    pdv_rp1=pdv_ref_p1,
                    pdv_rl2=pdv_ref_w2,
                    pdv_rp2=pdv_ref_p2,
                    pdv_tp=pdv_tar_p,
                    fr=x,
                    fc=y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    c_goal_x=c_goal_x,
                    c_goal_y=c_goal_y,
                    laser_energy=energymJ,
                    total_energy = energyT,
                    shot_time=shot_time,
                    notes=current_notes,
                    flag=0)
        return energymJ, energyT, pdv_tar_p, self.laser_triggered
    
    def shoot_or_skip_flyer_with_pdv_adjustment(self, x, y, goal_x, goal_y, c_goal_x, c_goal_y, move_start, waveplate_angle):
        # PDV check and adjustment for flyers with varied surface roughness
        pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = self.pdv.read_PDV()
        while not (-20 < float(pdv_tar_p) < 0) and float(pdv_ref_p1) <= 16.00:
            set_power = float(pdv_ref_p1) + abs((abs(float(pdv_tar_p))-20)) + 1
            self.pdv.set_ref_pow(set_power)
            print('Changing PDV ref power')
            set_start = time.time()
            time.sleep(1)
            pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = self.pdv.read_PDV()
            while float(pdv_ref_p1) != set_power:
                pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = self.pdv.read_PDV()
                print(f"Reference Wavelength 1: {pdv_ref_w1}, Power: {pdv_ref_p1}")
                print(f"Reference Wavelength 2: {pdv_ref_w2}, Power: {pdv_ref_p2}")
                print(f"Target Power: {pdv_tar_p}")
                time.sleep(1)
                if (-20 < float(pdv_tar_p) < 0) or float(pdv_ref_p1) >= 16.00:
                    set_time = time.time() - set_start
                    print(f'PDV set time (seconds): {set_time:0.2f}')
                    break
                elif not (-20 < float(pdv_tar_p) < 0) and float(pdv_ref_p1) >= 16.00:
                    set_time = time.time() - set_start
                    print('PDV saturated')
                    print(f'PDV set time (seconds): {set_time:0.2f}')
                    break

        
        pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = self.pdv.read_PDV()
        pdv_tar_p = float(pdv_tar_p)
        if -20 < pdv_tar_p < 0:
            ##########################
            # self.delay_generator.pulse_on() # Pulse on
            read_start = time.time()
            timeout = 4  # seconds
            initial_energy_val = self.power_meter.read_energy()
            current_energy_val = initial_energy_val
            while time.time() - read_start < timeout:
                time.sleep(0.1)  
                new_energy_val = self.power_meter.read_energy()
                if new_energy_val != current_energy_val:
                    current_energy_val = new_energy_val
                    break  
            # self.delay_generator.pulse_off() # Pulse off
            energymJ = current_energy_val * 1e3
            energymJ = round(energymJ, 2)
            energyT = 11.13 * energymJ + 31.59
            energyT = round(energyT, 2)
            print(f"Measured Reference Energy (mJ): {energymJ}")
            print(f"Measured Total Energy (mJ): {energyT}")
            #############################

            shot_time = time.time() - move_start
            shot_time = round(shot_time, 2)
            print(f'Total time for shot: {shot_time}')
            self.laser_triggered = True  # Set flag to prevent re-triggering

            # frame_a = self.top_camera.get_image()
            # cv2.imwrite(f'Images/frame_{x}_{y}_j.jpg', frame_a)

            current_notes = 'Sucessful laser shot'
            print('Sucessful laser shot')
            self.log_experiment_to_excel(
                    exp_type = self.exp_type,
                    flyer_id=self.flyer_id, 
                    flyer_material=self.flyer_material, 
                    flyer_thickness=self.flyer_thickness, 
                    sample_id=self.sample_id, 
                    sample_material=self.sample_material, 
                    spacing=self.spacing, 
                    waveplate_angle=waveplate_angle,
                    pdv_rl1=pdv_ref_w1,
                    pdv_rp1=pdv_ref_p1,
                    pdv_rl2=pdv_ref_w2,
                    pdv_rp2=pdv_ref_p2,
                    pdv_tp=pdv_tar_p,
                    fr=x,
                    fc=y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    c_goal_x=c_goal_x,
                    c_goal_y=c_goal_y,
                    laser_energy=energymJ,
                    total_energy = energyT,
                    shot_time=shot_time,
                    notes=current_notes,
                    flag=1)
        else:
            print('Skipping due to bad PDV return')
            shot_time = time.time() - move_start
            shot_time = round(shot_time, 2)
            current_notes = 'Skipping due to bad PDV return'
            self.laser_triggered = False
            energymJ = 0
            energyT = 0
            self.log_experiment_to_excel(
                    exp_type = self.exp_type,
                    flyer_id=self.flyer_id, 
                    flyer_material=self.flyer_material, 
                    flyer_thickness=self.flyer_thickness, 
                    sample_id=self.sample_id, 
                    sample_material=self.sample_material, 
                    spacing=self.spacing, 
                    waveplate_angle=waveplate_angle,
                    pdv_rl1=pdv_ref_w1,
                    pdv_rp1=pdv_ref_p1,
                    pdv_rl2=pdv_ref_w2,
                    pdv_rp2=pdv_ref_p2,
                    pdv_tp=pdv_tar_p,
                    fr=x,
                    fc=y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    c_goal_x=c_goal_x,
                    c_goal_y=c_goal_y,
                    laser_energy=energymJ,
                    total_energy = energyT,
                    shot_time=shot_time,
                    notes=current_notes,
                    flag=0)
        return energymJ, energyT, pdv_tar_p, self.laser_triggered
    
    def skip_flyer(self, x, y, goal_x, goal_y, move_time, waveplate_angle):
        print('Skipped due to failure in flyer detection')
        current_notes = 'Skipped due to failure in flyer detection'
        shot_time = round(move_time, 2)
        energymJ = 0
        energyT = 0
        self.log_experiment_to_excel(
                    exp_type = self.exp_type,
                    flyer_id=self.flyer_id, 
                    flyer_material=self.flyer_material, 
                    flyer_thickness=self.flyer_thickness, 
                    sample_id=self.sample_id, 
                    sample_material=self.sample_material, 
                    spacing=self.spacing, 
                    waveplate_angle=waveplate_angle,
                    pdv_rl1=0,
                    pdv_rp1=0,
                    pdv_rl2=0,
                    pdv_rp2=0,
                    pdv_tp=0,
                    fr=x,
                    fc=y,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    c_goal_x=0,
                    c_goal_y=0,
                    laser_energy=energymJ,
                    total_energy = energyT,
                    shot_time=shot_time,
                    notes=current_notes,
                    flag=0)

    def set_waveplate_angle(self, waveplate_angle):
        """Wavplate rotation"""
        if waveplate_angle != self.previous_waveplate_angle:
            self.waveplate_rotator.rotate(waveplate_angle)
            print(f"Waveplate rotated to {waveplate_angle} degrees.")
            self.previous_waveplate_angle = waveplate_angle
        else:
            print(f"Waveplate angle remains unchanged at {waveplate_angle} degrees.")

                
if __name__ == "__main__":
    controller = HelixController()
    controller.connect()