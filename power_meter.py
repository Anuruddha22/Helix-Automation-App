import serial
import serial.tools.list_ports
import time


class PowerMeter:
    """Class to handle Gentec-EO power meter over serial interface."""

    def __init__(self, com_port="COM3", verbose=False):
        self.com_port = com_port
        self.serial_port = None
        self.reply = ""
        self.custom_delay = 0.02
        self.verbose = verbose

        if self.com_port:
            self.connect(self.com_port)

    def list_ports(self):
        """Return a list of available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect(self, port=None):
        """Establish connection to power meter."""
        if port:
            self.com_port = port

        try:
            self.serial_port = serial.Serial(
                port=self.com_port,
                baudrate=115200,
                timeout=2
            )
            if self.verbose:
                print(f"[PowerMeter] Connected to {self.com_port}")
            return True
        except serial.SerialException as e:
            print(f"[PowerMeter] Serial connection error: {e}")
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            if self.verbose:
                print(f"[PowerMeter] Disconnected from {self.com_port}")

    def is_connected(self):
        """Check if serial port is open."""
        return self.serial_port and self.serial_port.is_open

    def send_command(self, command, arg=""):
        """Send command (and optional argument) to the power meter."""
        self.reply = ""
        full_cmd = command + arg
        for c in full_cmd:
            self.serial_port.write(c.encode("ascii"))
            time.sleep(self.custom_delay)

        if self.verbose:
            print(f"[PowerMeter] Sent: {full_cmd}")

    def read_line(self):
        """Read one line of response from the serial port."""
        self.reply = self.serial_port.readline().decode("ascii").strip()
        if self.verbose:
            print(f"[PowerMeter] Received: {self.reply}")
        return self.reply

    def read_fixed(self, length):
        """Read a fixed number of bytes."""
        self.reply = self.serial_port.read(length).decode("ascii")
        if self.verbose:
            print(f"[PowerMeter] Read fixed: {self.reply}")
        return self.reply

    def get_data(self):
        """ Read the line and convert the output into data """
        self.read_line()
        try:
            en = float(self.reply)
        except ValueError:
            en = 0.0
        return en
    
    def set_range_300(self):
        self.send_command('*SCS23') # Set range to 23: 300 millijoule, 25: 3 joule
        self.read_line()

    def set_range_100(self):
        self.send_command('*SCS22') # Set range to 22: 100 millijoule, 25: 3 joule
        self.read_line()

    def check_range(self):
        self.send_command('*GCR') # Check current range
        self.read_line()

    def read_energy(self):
        """Read energy from the power meter."""
        self.send_command("*CVU")  # Command to read energy in single-shot mode
        enMJ = self.get_data()
        return enMJ

if __name__ == "__main__":
    pm = PowerMeter("COM3")
    if pm.is_connected():
        pm.set_range()
        energy = pm.read_energy()
        energymJ = energy * 1e3  # Convert to mJ
        print(f"Measured energy: {energymJ:0.4f} mJ")
        pm.disconnect()
