import clr 
import time 

# Write in file paths of dlls needed. 
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.IntegratedStepperMotorsCLI.dll")

# Import functions from dlls. 
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from System import Decimal 


class WaveplateRotator:
    def __init__(self, serial_no="55481854"):
        self.serial_no = serial_no
        self.device = None

    def connect(self):
        """Connect and initialize the waveplate rotator device."""
        try:
            DeviceManagerCLI.BuildDeviceList()

            self.device = CageRotator.CreateCageRotator(self.serial_no)
            self.device.Connect(self.serial_no)

            if not self.device.IsSettingsInitialized():
                self.device.WaitForSettingsInitialized(10000)

            self.device.StartPolling(250)
            time.sleep(0.25)
            self.device.EnableDevice()
            time.sleep(0.25)

            self.device.LoadMotorConfiguration(self.serial_no, DeviceConfiguration.DeviceSettingsUseOptionType.UseDeviceSettings)

            info = self.device.GetDeviceInfo()
            print(f"Connected to: {info.Description}")

        except Exception as e:
            print(f"[Waveplate] Connection error: {e}")

    def home(self):
        """Home the device"""
        if self.device:
            print("[Waveplate] Homing device...")
            self.device.Home(60000)
            print("[Waveplate] Homing complete.")

    def rotate(self, waveplate_angle):
        """Move to an absolute waveplate angle in degrees."""
        if self.device:
            a = 0.0  # reference offset
            new_pos = Decimal(a + waveplate_angle)
            print(f"[Waveplate] Rotating to {new_pos} degrees...")
            self.device.MoveTo(new_pos, 60000)
            print("[Waveplate] Rotation complete.")

    def disconnect(self):
        """Cleanly disconnect the device."""
        if self.device:
            self.device.StopPolling()
            self.device.Disconnect()
            print("[Waveplate] Disconnected.")



if __name__ == "__main__":
    waveplate = WaveplateRotator()
    waveplate.connect()
    waveplate.home()  # Uncomment if needed
    waveplate.rotate(30.0)  # angle in degrees
    waveplate.disconnect()
