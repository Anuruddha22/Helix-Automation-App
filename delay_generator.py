import socket
import time
import logging
from constants import DDG_HOST, DDG_PORT, PULSE_TIME

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

class Delay_Generator:
  def __init__(self):
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  def connect(self):
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.connect((DDG_HOST, DDG_PORT))
    # TODO: check connection

  def disconnect(self):
    self.s.close()

  def send_command(self, command_str: str):
    _logger.info('Running command %s' % command_str)
    try:
      command = ('%s\r\n' % command_str).encode()
    except UnicodeEncodeError:
      # TODO
      pass
    self.s.sendall(command)
    data = self.s.recv(1024)  # TODO: enough bytes?
    try:
      data = data.decode().strip()
    except UnicodeDecodeError:
      # TODO
      pass

  def toggle_pulse(self):
    self.send_command(':PULSE0:STATE ON')
    time.sleep(PULSE_TIME)
    self.send_command(':PULSE0:STATE OFF')

  def pulse_on(self):
    self.send_command(':PULSE0:STATE ON')
  
  def pulse_off(self):
    self.send_command(':PULSE0:STATE OFF')

if __name__ == "__main__":
    ddg = Delay_Generator()
    ddg.connect()
    ddg.toggle_pulse()
    ddg.disconnect()