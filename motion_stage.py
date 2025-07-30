import SPiiPlusPython as sp
from constants import MS_HOST, MS_PORT, SAMPLE_HOME_X, SAMPLE_HOME_Y
import time

class MotionStage:
    def __init__(self):
        self.stage = sp.OpenCommEthernetTCP(MS_HOST, MS_PORT)
        if self.stage == -1:
            print("Failed to connect")
        else:
            print("Succesfully connected to the motion stage")

    def connect(self, motion_stage):
        motion_stage.enable_axis()
        motion_stage.set_velocity(50)
        
    def enable_axis(self):
        sp.Enable(self.stage, 0, sp.SYNCHRONOUS, True)
        print("Axis x has been enabled")
        sp.Enable(self.stage, 1, sp.SYNCHRONOUS, True)
        print("Axis y has been enabled")

    def disable_axis(self):
        sp.Disable(self.stage, 0, sp.SYNCHRONOUS, True)
        print("Axis x has been disabled")
        sp.Disable(self.stage, 1, sp.SYNCHRONOUS, True)
        print("Axis y has been disabled")

    def set_velocity(self, vel=50):
        sp.SetVelocity(self.stage, 0, vel, sp.SYNCHRONOUS, True)
        sp.SetVelocity(self.stage, 1, vel, sp.SYNCHRONOUS, True)
        print("Velocity for both axes was set to: ", vel)

    def run_home(self, home_x=SAMPLE_HOME_X, home_y=SAMPLE_HOME_Y):
        sp.ToPoint(self.stage, 0, 0, home_x, sp.SYNCHRONOUS, True) 
        sp.ToPoint(self.stage, 0, 1, home_y, sp.SYNCHRONOUS, True) 
        print(f"Moved to home position: ({home_x}, {home_y})")

    def move_to_point(self, x, y):
        sp.ToPoint(self.stage, 0, 0, round(x,4), sp.SYNCHRONOUS, True)
        sp.ToPoint(self.stage, 0, 1, round(y,4), sp.SYNCHRONOUS, True)
        print(f"Moved to position: ({x:.4f}, {y:.4f})")

    def get_position(self):
        x = sp.GetFPosition(self.stage, 0, sp.SYNCHRONOUS, True)
        y = sp.GetFPosition(self.stage, 1, sp.SYNCHRONOUS, True)
        return x, y
    
    def disconnect(self, motion_stage):
        motion_stage.disable_axis()
        sp.CloseComm(self.stage)
        print("Connection closed")
    


if __name__ == "__main__":
    stage = MotionStage()
    stage.connect(stage)
    time.sleep(1)
    stage.run_home()
    time.sleep(2)
    stage.disconnect(stage)