from ids_peak import ids_peak as peak
from ids_peak import ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl
import cv2
import numpy as np
import logging
import time 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
_logger = logging.getLogger(__name__)

class Top_Camera:
  def __init__(self):
    """
    Initialize the IDS camera using the IDS peak API.
    """
    # initialize library
    peak.Library.Initialize()
    try:
        # Create instance of the device manager
        device_manager = peak.DeviceManager.Instance()
        # Update the device manager
        device_manager.Update()
        # Return if no device was found
        if device_manager.Devices().empty():
            return False
        # open the first openable device in the device manager's device list
        #TODO: Update code to select the top camera ID specifically to avoid other camera
        device_count = device_manager.Devices().size()
        for i in range(device_count):
            if device_manager.Devices()[i].IsOpenable():
                self.device = device_manager.Devices()[i].OpenDevice(peak.DeviceAccessType_Control)
                # Get NodeMap of the RemoteDevice for all accesses to the GenICam NodeMap tree
                self.node_map_device = self.device.RemoteDevice().NodeMaps()[0]
        _logger.info('Camera initialized successfully.')

    except Exception as e:
    # ...
       logging.error(f"An error occurred during the camera connection: {e}")

  def prepare_acquisition(self):
     try:
       data_streams = self.device.DataStreams()
       if data_streams.empty():
          # no data streams available
          return False
       self.dataStream = self.device.DataStreams()[0].OpenDataStream()

     except Exception as e:
      # ...
       str_error = str(e)
     return self.dataStream

  def set_roi(self, x, y, width, height):
    try:
      # Get the minimum ROI and set it. After that there are no size restrictions anymore
       x_min = self.node_map_device.FindNode("OffsetX").Minimum()
       y_min = self.node_map_device.FindNode("OffsetY").Minimum()
       w_min = self.node_map_device.FindNode("Width").Minimum()
       h_min = self.node_map_device.FindNode("Height").Minimum()

       self.node_map_device.FindNode("OffsetX").SetValue(x_min)
       self.node_map_device.FindNode("OffsetY").SetValue(y_min)
       self.node_map_device.FindNode("Width").SetValue(w_min)
       self.node_map_device.FindNode("Height").SetValue(h_min)

      # Get the maximum ROI values
       x_max = self.node_map_device.FindNode("OffsetX").Maximum()
       y_max = self.node_map_device.FindNode("OffsetY").Maximum()
       w_max = self.node_map_device.FindNode("Width").Maximum()
       h_max = self.node_map_device.FindNode("Height").Maximum()

       if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
           return False
       elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
           return False
       else:
          # Now, set final AOI
           self.node_map_device.FindNode("OffsetX").SetValue(x)
           self.node_map_device.FindNode("OffsetY").SetValue(y)
           self.node_map_device.FindNode("Width").SetValue(width)
           self.node_map_device.FindNode("Height").SetValue(height)
           return True
       
    except Exception as e:
        # ...
        str_error = str(e)

  def alloc_and_announce_buffers(self):
    try:
        if self.dataStream:
            # Flush queue and prepare all buffers for revoking
            self.dataStream.Flush(peak.DataStreamFlushMode_DiscardAll)
            # Clear all old buffers
            for buffer in self.dataStream.AnnouncedBuffers():
                self.dataStream.RevokeBuffer(buffer)
            payload_size = self.node_map_device.FindNode("PayloadSize").Value()
            # Alloc buffers
            bufferCountMax = self.dataStream.NumBuffersAnnouncedMinRequired()
            for bufferCount in range(bufferCountMax):
                buffer = self.dataStream.AllocAndAnnounceBuffer(payload_size)
                self.dataStream.QueueBuffer(buffer)
        
    except Exception as e:
        # ...
        logging.error(f"An error occurred during allocating buffer: {e}")

  def load_set_camera_settings(self):
    """
    Load camera settings from a .cset file using the IDS peak API. 
    We could also use constants and set those directly for different settings.
    """
    # Determine the current entry of UserSetSelector (str)
    value = self.node_map_device.FindNode("UserSetSelector").CurrentEntry().SymbolicValue()
    print('Current UserSetSelector: ', value)
    # Get a list of all available entries of UserSetSelector
    allEntries = self.node_map_device.FindNode("UserSetSelector").Entries()
    availableEntries = []
    for entry in allEntries:
        print('all entries', entry.SymbolicValue())
        if (entry.AccessStatus() != peak.NodeAccessStatus_NotAvailable
                and entry.AccessStatus() != peak.NodeAccessStatus_NotImplemented):
            availableEntries.append(entry.SymbolicValue())
        print('Access status', entry.AccessStatus())
    print('Avaiable UserSetSelector: ', availableEntries)
    try:
        # file contains the fully qualified path to the file
        file = "support_files/4108570910.cset"
        # Load from file
        self.node_map_device.LoadFromFile(file)

    except Exception as e:
        print("Exception: " + str(e) + "")

  def set_acquisition_mode(self):
    # prepare for untriggered continuous image acquisition
    self.node_map_device.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
    self.node_map_device.FindNode("TriggerMode").SetCurrentEntry("Off")
    self.node_map_device.FindNode("TriggerSource").SetCurrentEntry("Software")
    # Set gain mode
    self.node_map_device.FindNode('GainSelector').SetCurrentEntry('DigitalAll')
    # Set gain constant
    # self.node_map_device.FindNode('Gain').SetValue(8.0)


  def start_acquisition(self):
    try:
        # self.dataStream.StartAcquisition(peak.AcquisitionStartMode_Default, peak.DataStream.INFINITE_NUMBER)
        self.dataStream.StartAcquisition()
        # self.node_map_device.FindNode("TLParamsLocked").SetValue(1)
        self.node_map_device.FindNode("AcquisitionStart").Execute()
        self.node_map_device.FindNode("TriggerSoftware").Execute()

    except Exception as e:
        # ...
        str_error = str(e)

  def capture_and_save_image(self):
    # get buffer from datastream 
    buffer = self.dataStream.WaitForFinishedBuffer(5000)
    if buffer:
        raw_image = ids_peak_ipl_extension.BufferToImage(buffer)
        color_image = raw_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
        # conver to numpy array
        np_image = color_image.get_numpy_3D()
        np_resize = cv2.resize(np_image, (480 * 2, 480 * 2))
        frame = np_resize[:, 160:1120] #cropping
        cv2.imshow('image', frame)
        cv2.waitKey(1)
        # save the captured image.
        cv2.imwrite('Images/captured_image.jpg', frame)
        # queue buffer
        self.dataStream.QueueBuffer(buffer)
    else:
        _logger.info('The buffer from Datastream is empty')

    return frame
  
  def get_image(self):
    # get buffer from datastream 
    buffer = self.dataStream.WaitForFinishedBuffer(5000)
    if buffer:
        raw_image = ids_peak_ipl_extension.BufferToImage(buffer)
        color_image = raw_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
        # conver to numpy array
        np_image = color_image.get_numpy_3D()
        # cropped_image = np_image[1382:2342, 2386:3346]
        offsetx =  2298 #2262
        offsety = 1364 # 1350
        width = offsety + 960
        height = offsetx + 960
        cropped_image = np_image[offsety:width, offsetx:height]
        frame = cv2.resize(cropped_image, (960, 960))
        # np_resize = cv2.resize(np_image, (640 * 2, 480 * 2))
        # frame = np_resize[:, 160:1120] #cropping
        # frame = np_resize[:, 160:1120]
        # cv2.imshow('image', frame)
        # cv2.waitKey(1)
        # queue buffer
        self.dataStream.QueueBuffer(buffer)
    else:
        _logger.info('The buffer from Datastream is empty')

    return frame
  
def disconnect_camera():
    peak.Library.Close()
    _logger.info('Camera is closed')

def run_camera_workflow(top_camera):
    """
    Full workflow to initialize the camera, load settings, capture, and save an image.
    """
    try:
        # prepare acquisation
        top_camera.prepare_acquisition()

        # # Set ROI
        # top_camera.set_roi(16, 16, 600, 600)

        # Allocate buffers for capturing images
        top_camera.alloc_and_announce_buffers()

        # Load and set camera settings 
        top_camera.load_set_camera_settings()

        # start acquisation mode
        top_camera.set_acquisition_mode()
        top_camera.start_acquisition()
        # top_camera.capture_and_save_image()

    except Exception as e:
        _logger.error(f"An error occurred during the camera workflow: {e}")
    

    
if __name__ == "__main__":
    # Example usage
    top_camera = Top_Camera()
    if top_camera:
        run_camera_workflow(top_camera)
        time.sleep(1)
        image = top_camera.get_image()
        cv2.imshow('Captured Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Close camera library
        disconnect_camera() 
    else:
        _logger.error("Failed to initialize the camera.")
   