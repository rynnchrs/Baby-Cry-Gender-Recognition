from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time
start = time.time()

label = []
features = []
data = []

decision_tree_pkl_filename = "boy_vs_girl_knn_fft_model.pkl"

fs_rate, signal = wavfile.read('/home/rynnchrs/Desktop/Baby_Cry_Gender_Recognition/Dataset/BOY/boy1.wav')
print ("Frequency sampling", fs_rate)
signal = np.resize(signal,(130000,2))
l_audio = len(signal.shape)
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print ("Complete Samplings N", N)
secs = N / float(fs_rate)
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
t = np.resize(t,(130000,))
FFT = abs(scipy.fft(signal))
FFT_side = FFT[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)

data.append(FFT)

decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)

print("Predicted Gender: " + str(decision_tree_model.predict(data)))


end = time.time()
print("Time Response: " + str(end - start) + "s")

plt.subplot(311)
p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(313)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()