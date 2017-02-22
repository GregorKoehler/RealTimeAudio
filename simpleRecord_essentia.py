import pyaudio
import numpy as np
from essentia.standard import *

CHUNK = 4096 # number of data points to read at a time
RATE = 44100 # time resolution of the recording device (Hz)

p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

# some essentia algorithms
w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()


# create a numpy array holding a single read of audio data
for i in range(10): #to it a few times just to see
    data = np.fromstring(stream.read(CHUNK),dtype=np.float32)
    print(data)
    print data.shape
    print data.dtype
    spec = spectrum(w(data))
    print spec
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(data)))
    print mfcc_coeffs




# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()
