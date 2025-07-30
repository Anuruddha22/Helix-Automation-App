import sys
import cv2
import numpy as np
import time
import random
import traceback
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QDialog, QLineEdit, QTextEdit, QFormLayout, QGridLayout, QListWidget, QListWidgetItem,
                             QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from matplotlib.figure import Figure
from helix_controller import HelixController

# ----------------------------- Circle Cell Widget (Drawn Circle) -----------------------------
class CircleCell(QWidget):
    def __init__(self, color='gray'):
        super().__init__()
        self.color = color
        self.setFixedSize(24, 24)

    def set_color(self, color):
        self.color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(self.color))
        painter.setPen(Qt.black)
        painter.drawEllipse(2, 2, 20, 20)

# ----------------------------- Dialog to Input Parameters -----------------------------
class ExperimentParameterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter Experiment Parameters")
        layout = QFormLayout()

        self.exp_type = QLineEdit("calibration shots")
        self.flyer_id = QLineEdit("F399")
        self.sample_id = QLineEdit("S530")
        self.flyer_material = QLineEdit("Al")
        self.sample_material = QLineEdit("Cu")
        self.flyer_thickness = QLineEdit("100")
        self.spacing = QLineEdit("600")
        self.waveplate_angles = QLineEdit("25, 27, 30") 
        self.pdv_file_number = QLineEdit("00005")
        self.beam_file_number = QLineEdit("00005")
        self.notes = QTextEdit("App code test")

        layout.addRow("Experiment Type:", self.exp_type)
        layout.addRow("Flyer ID:", self.flyer_id)
        layout.addRow("Sample ID:", self.sample_id)
        layout.addRow("Flyer Material:", self.flyer_material)
        layout.addRow("Sample Material:", self.sample_material)
        layout.addRow("Flyer Thickness (um):", self.flyer_thickness)
        layout.addRow("Spacing (um):", self.spacing)
        layout.addRow("Waveplate Angles (Degrees):", self.waveplate_angles) 
        layout.addRow("PDV File Number:", self.pdv_file_number)
        layout.addRow("Beam Profile File Number:", self.beam_file_number)
        layout.addRow("Notes:", self.notes)

        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.accept)
        layout.addRow(self.done_button)
        self.setLayout(layout)

    def get_parameters(self):
        return {
            "exp_type": self.exp_type.text(),
            "flyer_id": self.flyer_id.text(),
            "sample_id": self.sample_id.text(),
            "flyer_material": self.flyer_material.text(),
            "sample_material": self.sample_material.text(),
            "flyer_thickness": self.flyer_thickness.text(),
            "spacing": self.spacing.text(),
            "waveplate_angle": self.waveplate_angles.text(),
            "pdv_file_number": self.pdv_file_number.text(),
            "beam_file_number": self.beam_file_number.text(),
            "notes": self.notes.toPlainText()
        }

# ----------------------------- Circle Grid Widget -----------------------------
class CircleGrid(QWidget):
    def __init__(self, selected):
        super().__init__()
        self.selected = selected
        self.finished = []
        self.skipped = []
        self.skipped_tracking = []
        self.skipped_pdv = []
        self.grid = QGridLayout()
        self.buttons = {}
        self.setLayout(self.grid)
        for i in range(7):
            for j in range(7):
                cell = CircleCell()
                self.grid.addWidget(cell, i, j)
                self.buttons[(i, j)] = cell

        self.update_grid()

    def update_grid(self):
        for (i, j), cell in self.buttons.items():
            if (i, j) in self.skipped_tracking:
                cell.set_color('red')  # Skipped_tracking = red
            elif (i, j) in self.skipped_pdv:
                cell.set_color('black')  # Skipped_pdv = black
            elif (i, j) in self.finished:
                cell.set_color('green')  # Done = green
            elif (i, j) in self.selected:
                cell.set_color('yellow')  # Selected = yellow
            else:
                cell.set_color('lightgray')  # Default


    def mark_done(self, i, j, skipped_pdv=False, skipped_tracking=False):
        if skipped_pdv:
            if (i, j) not in self.skipped_pdv:
                self.skipped_pdv.append((i, j))
        elif skipped_tracking: 
            if (i, j) not in self.skipped_pdv:
                self.skipped_tracking.append((i, j))
        else:
            if (i, j) not in self.finished:
                self.finished.append((i, j))
        self.update_grid()

# ----------------------------- Plotting Widget -----------------------------
class LivePlot(FigureCanvas):
    def __init__(self, reference_excel_file):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title("Laser Energy vs Waveplate Angle")
        self.ax.set_xlabel("Waveplate Angle (deg)")
        self.ax.set_ylabel("Laser Energy (mJ)")
        self.ax.grid(True)

        # Load and process reference data
        df = pd.read_excel(reference_excel_file)
        grouped = df.groupby('Waveplate')['Reference Energy (mJ)'].mean().reset_index()

        self.ref_angles = grouped['Waveplate'].tolist()
        self.ref_energies = grouped['Reference Energy (mJ)'].tolist()
        self.ax.plot(self.ref_angles, self.ref_energies, color='black', marker='o', linestyle='dashed', label='Reference (Calibrated Mean)', linewidth=0.01, markersize=5)


        # Prepare experimental plot
        self.exp_angles = []
        self.exp_energies = []
        self.exp_line, = self.ax.plot([], [], 'ro', label='Experimental (Live)', markersize=5)

        self.ax.legend()
        self.ax.set_xlim(min(self.ref_angles) - 5, max(self.ref_angles) + 5)
        self.ax.set_ylim(0, max(self.ref_energies) + 100)

    def update_plot(self, angle, energy):
        self.exp_angles.append(angle)
        self.exp_energies.append(energy)
        self.exp_line.set_data(self.exp_angles, self.exp_energies)

        self.ax.set_xlim(min(self.ref_angles + self.exp_angles) - 5,
                         max(self.ref_angles + self.exp_angles) + 5)
        self.ax.set_ylim(0, max(self.ref_energies + self.exp_energies) + 100)

        self.draw()

# ----------------------------- Final GUI with Live Feed -----------------------------

class LiveFeedApp(QWidget):
    def __init__(self, selected_circles, parameters):
        super().__init__()
        self.setWindowTitle("HELIX App")
        self.selected_circles = selected_circles
        self.parameters = parameters
        self.current_index = 0
        self.experiment_running = False

        # ---------- HELIX Controller Integration ----------
        self.controller = HelixController()
        self.controller.connect()
        try:
            self.controller.load_parameters(parameters)
            self.waveplate_angle = self.controller.waveplate_angle_list
        except Exception as e:
            print(f"ERROR: Failed to load controller parameters: {e}")
            self.status_label.setText("Status: Failed to load parameters")
        # ---------------------------------------------------

        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 480)

        self.info_label = QLabel("System Var: --")
        self.info_label.setStyleSheet("font-size: 14px;")

        self.plot_widget = LivePlot("support_files/angle_vs_energy_calibration_full.xlsx")

        self.grid_widget = CircleGrid(self.selected_circles)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: blue;")

        self.param_box = QTextEdit()
        self.param_box.setReadOnly(True)
        param_text = "\n".join(f"{k}: {v}" for k, v in parameters.items())
        self.param_box.setText(param_text)
        self.param_box.setFixedHeight(120)

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color: green; color: white;")
        self.start_button.clicked.connect(self.start_experiment)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: red; color: white;")
        self.stop_button.clicked.connect(self.stop_experiment)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.image_label)
        top_right_layout = QVBoxLayout()
        top_right_layout.addWidget(self.plot_widget)
        top_right_layout.addWidget(self.grid_widget)
        top_layout.addLayout(top_right_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.param_box)
        self.setLayout(main_layout)

        self.timer_image = QTimer()
        self.timer_image.timeout.connect(self.update_image)
        self.timer_image.start(30)

    def start_experiment(self):
        self.experiment_running = True

        self.status_label.setText("Status: Running experiment...")
        QApplication.processEvents()
        
        for x, y in self.selected_circles:
            if not self.experiment_running:
                break
            
            self.status_label.setText("Status: Setting up waveplate angle...")
            QApplication.processEvents()

            try:
                index = self.selected_circles.index((x, y))
            except ValueError:
                index = 0

            # set waveplate angle based on index
            if len(self.controller.waveplate_angle_list) == 1:
                self.waveplate_angle = self.controller.waveplate_angle_list[0]
            elif index < len(self.controller.waveplate_angle_list):
                self.waveplate_angle = self.controller.waveplate_angle_list[index]
            else:
                self.waveplate_angle = self.controller.waveplate_angle_list[-1]

            self.controller.set_waveplate_angle(self.waveplate_angle)

            if self.waveplate_angle > 35:
                self.controller.power_meter.set_range_100()
            if 15 < self.waveplate_angle < 35:
                self.controller.power_meter.set_range_300()
            if self.waveplate_angle <= 15:
                self.controller.power_meter.set_range_1000()

            # # For PDV power adjustment only
            # self.controller.pdv.set_ref_pow(10.00)
            # print('PDV ref power reset to 10 dB')
            # time.sleep(10)

            self.status_label.setText(f"Status: Moving to ({x+1}, {y+1})")
            QApplication.processEvents()
            time.sleep(1)
            goal_x, goal_y = self.controller.move_initial(x, y)
            self.is_moving = True
            self.laser_triggered = False
            self.skipped_flyer = False
            self.skipped_flyer_pdv = False
            self.skipped_flyer_tracking = False
            laser_energy = 0.0
            total_energy = 0.0
            move_start = time.time()

            while True: 
                self.pause_live_feed()

                self.status_label.setText("Status: Detecting flyer...")
                QApplication.processEvents()

                print("Calling detect_flyer()...")
                frame, center = self.controller.detect_flyer()
                print("Returned from detect_flyer()")


                if frame is None or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    print("Invalid frame received from detect_flyer.")
                    self.status_label.setText("Status: Invalid image detected, skipping shot.")
                    QApplication.processEvents()
                    continue  
                else:
                    print('Frame is valid')
                    try:
                        print('Trying to update image')
                        self.update_gui_image(frame)
                        print("Image successfully updated in GUI.")
                    except Exception as e:
                        print("Error while updating GUI image:", e)
                        traceback.print_exc()
                        self.status_label.setText("Status: Failed to update image.")
                        QApplication.processEvents()
                        continue

                self.resume_live_feed()

                move_time = time.time() - move_start
                if self.is_moving and not self.laser_triggered and move_time > 1:
                    self.status_label.setText("Status: Aligning flyer with the laser...")
                    QApplication.processEvents()
                    c_goal_x, c_goal_y = self.controller.align_flyer(goal_x, goal_y, center)
                    self.is_moving = False

                move_time = time.time() - move_start
                if not self.laser_triggered and not self.is_moving and move_time > 2 and ((abs(480 - center[0]) < 15 and abs(480 - center[1]) < 15)):
                    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)  # Mark the circle's center
                    self.update_gui_image(frame)
                    move_time = time.time() - move_start
                    if 20 > move_time > 3:
                        self.status_label.setText("Status: Shooting laser or skipping...")
                        QApplication.processEvents()
                        self.timer_image.start(30)
                        # For shooting without PDV power adjustment
                        laser_energy, total_energy, pdv_vals, self.laser_triggered = self.controller.shoot_or_skip_flyer_with_pdv_reading(x, y, goal_x, goal_y, c_goal_x, c_goal_y, move_start, self.waveplate_angle)
                        # For shooting with PDV power adjustment
                        # Uncomment the line below to use PDV power adjustment and comment the above method
                        # laser_energy, total_energy, pdv_vals, self.laser_triggered = self.controller.shoot_or_skip_flyer_with_pdv_adjustment(x, y, goal_x, goal_y, c_goal_x, c_goal_y, move_start, self.waveplate_angle)
                        
                        self.update_variable(pdv_vals, total_energy)
                        if not self.laser_triggered:
                            self.skipped_flyer_pdv = True
                            self.status_label.setText("Status: Skipped due to bad PDV return...")
                            QApplication.processEvents()
                            break
                        if self.laser_triggered:
                            self.skipped_flyer_pdv = False
                            self.skipped_flyer_tracking = False
                            self.plot_widget.update_plot(self.waveplate_angle, total_energy)
                            self.status_label.setText("Status: Successful shot...")
                            QApplication.processEvents()
                            break
                
                move_time = time.time() - move_start
                if not self.laser_triggered and move_time >= 20:
                    self.controller.skip_flyer(x, y, goal_x, goal_y, move_time, self.waveplate_angle)
                    self.skipped_flyer_tracking = True
                    self.status_label.setText("Status: Skipped due to failed flyer detection...")
                    QApplication.processEvents()
                    break
            
            move_time = time.time() - move_start
            self.status_label.setText(f"Done ({x+1},{y+1}) | Target Energy: {total_energy:.2f} mJ")
            QApplication.processEvents()
            self.grid_widget.mark_done(x, y, skipped_pdv=self.skipped_flyer_pdv, skipped_tracking=self.skipped_flyer_tracking)

        self.status_label.setText("Status: Experiment completed.")
        self.experiment_running = False


    def stop_experiment(self):
        if self.timer_image.isActive():
                self.timer_image.stop()  # pause live feed during experiment
        try:
            self.controller.disconnect()
        except Exception as e:
            print(f"Error during disconnection: {e}")
        self.status_label.setText("Status: Experiment stopped manually.")

    def update_image(self):
        frame = self.controller.top_camera.get_image()
        frame = cv2.resize(frame, (480, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def update_gui_image(self, frame):
        frame = cv2.resize(frame, (480, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.copy()  # to prevent buffer issues

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))



    def update_variable(self, pdv_val, laser_energy):
        self.info_label.setText(f"Waveplate Angle: {self.waveplate_angle}°, PDV Return: {pdv_val} dB, Laser Energy: {laser_energy:.2f} mJ")

    def pause_live_feed(self):
        if self.timer_image.isActive():
            self.timer_image.stop()

    def resume_live_feed(self):
        if not self.timer_image.isActive():
            self.timer_image.start(30)


    def closeEvent(self, event):
        event.accept()


# ----------------------------- Circlar Flyer Selection -----------------------------
class CircleSelectionDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Circular Flyers to Shoot")
        self.selected_circles = []
        layout = QHBoxLayout()
        grid_layout = QGridLayout()
        self.buttons = {}
        for i in range(7):
            for j in range(7):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                btn.setStyleSheet("""
                    background-color: white;
                    border: 1px solid black;
                    border-radius: 20px;
                """)
                btn.clicked.connect(self.make_toggle(i, j, btn))
                grid_layout.addWidget(btn, i, j)
                self.buttons[(i, j)] = btn
        self.selection_list = QListWidget()
        layout.addLayout(grid_layout)
        layout.addWidget(self.selection_list)
        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.accept)
        layout.addWidget(done_btn)
        self.setLayout(layout)

    def make_toggle(self, i, j, button):
        def toggle():
            if (i, j) in self.selected_circles:
                self.selected_circles.remove((i, j))
                button.setStyleSheet("""
                    background-color: white;
                    border: 1px solid black;
                    border-radius: 20px;
                """)
            else:
                self.selected_circles.append((i, j))
                button.setStyleSheet("""
                    background-color: yellow;
                    border: 1px solid black;
                    border-radius: 20px;
                """)
            self.update_list()
        return toggle

    def update_list(self):
        self.selection_list.clear()
        for idx, (x, y) in enumerate(self.selected_circles):
            self.selection_list.addItem(QListWidgetItem(f"{idx+1}: Circle ({x+1}, {y+1})"))

    def get_selected(self):
        return self.selected_circles

# ----------------------------- App Entry -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    param_dialog = ExperimentParameterDialog()
    if param_dialog.exec_():
        parameters = param_dialog.get_parameters()
        circle_dialog = CircleSelectionDialog()
        if circle_dialog.exec_():
            selected_circles = circle_dialog.get_selected()
            window = LiveFeedApp(selected_circles, parameters)
            window.resize(1024, 900)
            window.show()
            sys.exit(app.exec_())
