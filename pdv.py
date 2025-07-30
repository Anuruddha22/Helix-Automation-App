import vxi11
from vxi11.vxi11 import Vxi11Exception
import time
from constants import PDV_HOST

class PDV:
    def __init__(self, host=PDV_HOST):
        self.host = host
        self.instrument = None
        self.connect()
    
    def connect(self):
        try:
            print("connecting to " + self.host + " ... ")
            self.instrument = vxi11.Instrument(self.host)
            print("connected")
            print("checking IDN...")
            command = "*IDN?"
            data = self.instrument.ask(command)
            print("IDN: " + data)
            print("checking OPT...")
            command = "*OPT?"
            data = self.instrument.ask(command)
            print("OPT: " + data)
        except Vxi11Exception as e:
            print("ERROR" + str(e) + ", command: " + str(command))

    def turn_on(self):
        """ Turn on the PDV instrument """
        if self.instrument:
            command = ":OUTP2:CHAN1:STATE ON", ":OUTP2:CHAN2:STATE ON"
            self.instrument.write(command)
            print("PDV turned on")
        else:
            print("Could not turn on PDV, no connection established")

    def read_PDV(self):
        try:
            # Read reference and target wavelengths and powers
            ref_wav_input_c = ":SOUR2:CHAN1:WAV?", ":SOUR2:CHAN2:WAV?"
            ref_wav_input_r = self.instrument.ask(ref_wav_input_c)
            ref_pow_input_c = ":SOUR2:CHAN1:POW?", ":SOUR2:CHAN2:POW?"
            ref_pow_input_r = self.instrument.ask(ref_pow_input_c)
            tar_pow_c = ":OUTP6:CHAN1:POW?"
            tar_pow_r = self.instrument.ask(tar_pow_c)

            # Return variables
            ref_wav_1 = ref_wav_input_r[0]
            ref_wav_2 = ref_wav_input_r[1]
            ref_pow_1 = ref_pow_input_r[0]
            ref_pow_2 = ref_pow_input_r[1]
            tar_pow = tar_pow_r

        except Vxi11Exception as e:
            # pass
            print("ERROR" + str(e) + ", command: " + str(ref_wav_input_c) + " or " + str(ref_pow_input_c) + " or " + str(tar_pow_c))
        return  ref_wav_1, ref_pow_1, ref_wav_2, ref_pow_2, tar_pow

    def turn_off(self):
        """ Turn off the PDV instrument """
        if self.instrument:
            command = ":OUTP2:CHAN1:STATE OFF", ":OUTP2:CHAN2:STATE OFF"
            self.instrument.write(command)
            print("PDV turned off")
        else:
            print("Could not turn off PDV, no connection established")
    
    def set_ref_pow(self, set_pow= 7.50):
        try:
            ref_pow_set_c = f":SOUR2:CHAN1:POW {set_pow}"
            ref_pow_set_r = self.instrument.write(ref_pow_set_c)
        except Vxi11Exception as e:
            # pass
            print("ERROR" + str(e) + ", command: " + str(ref_pow_set_c))
        return  ref_pow_set_r



if __name__ == "__main__":
    pdv = PDV()
    # pdv.turn_on()
    pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = pdv.read_PDV()
    print(f"Reference Wavelength 1: {pdv_ref_w1}, Power: {pdv_ref_p1}")
    print(f"Reference Wavelength 2: {pdv_ref_w2}, Power: {pdv_ref_p2}")
    print(f"Target Power: {pdv_tar_p}")
    if not (-20 < float(pdv_tar_p) < 0):
        set_power = pdv_ref_p1 + 1
        pdv.set_ref_pow(set_power)
        set_start = time.time()
        time.sleep(1)
        pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = pdv.read_PDV()
        print(f"Reference Wavelength 1: {pdv_ref_w1}, Power: {pdv_ref_p1}")
        print(f"Reference Wavelength 2: {pdv_ref_w2}, Power: {pdv_ref_p2}")
        print(f"Target Power: {pdv_tar_p}")
        while pdv_ref_p1 != set_power:
            pdv_ref_w1, pdv_ref_p1, pdv_ref_w2, pdv_ref_p2, pdv_tar_p = pdv.read_PDV()
            print(f"Reference Wavelength 1: {pdv_ref_w1}, Power: {pdv_ref_p1}")
            print(f"Reference Wavelength 2: {pdv_ref_w2}, Power: {pdv_ref_p2}")
            print(f"Target Power: {pdv_tar_p}")
            time.sleep(1)
            if (-20 < float(pdv_tar_p) < 0) or float(pdv_ref_p1)==16:
                set_time = time.time() - set_start
                print(f'PDV set time: {set_time:0.2f}')
                break
    # pdv.turn_off()