import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import clr
from pathlib import Path
from PIL import Image

# === Configure this path ===
DLL_PATH = r"HelixApp/support_files/BeamageSDK.dll"

# === Preload DLL and import C# namespace ===
dll_dir = os.path.dirname(DLL_PATH)
os.environ["PATH"] += os.pathsep + dll_dir
clr.AddReference(DLL_PATH)
from BeamageApi import BSDK


class BeamageWrapper:
    def __init__(self):
        self.sdk = BSDK()
        self.sdk.SetCanvas(2048)
        self.sdk.Detect()

        if self.sdk.cameras.Count == 0:
            raise RuntimeError("No Beamage camera detected.")

        self.cam = self.sdk.cameras[0]
        print(f'Connecting camera with serial number: {self.cam.Properties.GetSerialNumber()}')
        print(f'Is Beamage-4M model? : {self.cam.Properties.Is4mSensor()}')
        self.cam.Connect()

    def run(self):
        self.cam.Run()

    def stop(self):
        self.cam.Stop()

    def grab_image(self):
        self.cam.GrabOneFrame()
        width = self.cam.Image.width
        height = self.cam.Image.height
        raw = self.cam.Image.GetImage()
        np_im = np.array(list(raw), dtype=np.uint16).reshape((height, width))
        bmp_im = self.cam.Image.GetBmpRealColor()
        return np_im, bmp_im

    def get_beam_metrics(self):
        dia_x = self.cam.Image.DiameterInfo.diameter4SigmaX
        dia_y = self.cam.Image.DiameterInfo.diameter4SigmaY
        cent_x = self.cam.Image.CentroidInfo.centroidXPos
        cent_y = self.cam.Image.CentroidInfo.centroidYPos
        return dia_x, dia_y, cent_x, cent_y

    def set_exposure(self, ms: float):
        self.cam.SetToAutoExposure(False)
        self.cam.SetExposureTime(ms)

    def enable_auto_exposure(self):
        self.cam.SetToAutoExposure(True)

    def capture_for_duration(
        self, duration_sec=5, interval_sec=0.1,
        save_dir=None, save_csv=True, display=True):
        print(f"Capturing for {duration_sec} seconds...")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            img_dir = save_dir / "frames"
            img_dir.mkdir(exist_ok=True)
            if save_csv:
                csv_path = save_dir / "beam_metrics.csv"
                csv_file = open(csv_path, "w", newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "time (s)", "4SigX", "4SigY", "centroid X", "centroid Y"])

        self.run()
        images = []
        timestamps = []
        metrics_log = []
        start = time.time()

        plt.ion()
        fig, ax = plt.subplots()
        im = None

        frame_idx = 0
        while time.time() - start < duration_sec:
            img_np, img_bmp = self.grab_image()
            img = Image.open(img_bmp)
            dia_x, dia_y, cent_x, cent_y = self.get_beam_metrics()
            t_now = time.time() - start

            images.append(img_np)
            timestamps.append(t_now)
            metrics_log.append((dia_x, dia_y, cent_x, cent_y))

            if save_dir:
                # img.save(img_dir / f"frame_{frame_idx:04d}.bmp", "bmp")
                np.save(img_dir / f"frame_{frame_idx:04d}.npy", img_np)
                if save_csv:
                    csv_writer.writerow([frame_idx, t_now, dia_x, dia_y, cent_x, cent_y])

            if display:
                if im is None:
                    im = ax.imshow(img_np, cmap='gray')
                    fig.colorbar(im, ax=ax)
                else:
                    im.set_data(img_np)
                ax.set_title(f"Frame {frame_idx} - {t_now:.2f}s")
                plt.pause(0.001)

            frame_idx += 1
            time.sleep(interval_sec)

        self.stop()
        if save_dir and save_csv:
            csv_file.close()
        plt.ioff()
        plt.show()

        print(f"Captured {frame_idx} frames.")
        return images, timestamps, metrics_log

    def shutdown(self):
        self.cam.Dispose()
        self.sdk.Dispose()


if __name__ == "__main__":
    profiler = BeamageWrapper() # 298906
    profiler.set_exposure(20)  # milliseconds # profiler.enable_auto_exposure()

    images, times, metrics = profiler.capture_for_duration(
        duration_sec=5,
        interval_sec=0.1,
        save_dir="beamage_output",
        display=False
    )

    profiler.shutdown()