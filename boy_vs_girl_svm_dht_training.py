# import libraries
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import os
import pickle

# use array for data manipulation
svmdata=[]
label = []

# looping of the dataset folder
folder = r"D:\Projects\Thesis\Dataset\BOY"
for filename in os.listdir(folder):

    # reading wav file in the dataset
    fs_rate, signal = wavfile.read(os.path.join(folder,filename))
    print ("Frequency sampling", fs_rate)

    # resizing the signal length
    signal = np.resize(signal,(130000,2))
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)

    # sampling interval in time
    Ts = 1.0/fs_rate
    print ("Timestep between samples Ts", Ts)

    # time vector as scipy arange field / numpy.ndarray
    t = np.arange(0, secs, Ts)
    t = np.resize(t,(130000,))
    FFT = abs(scipy.fft.fft(signal))
    FHT = scipy.fft.fft(signal) * np.array([1 + 1j])

    # one side FFT range
    FFT_side = FFT[range(N//2)]
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)]

    # one side frequency range
    fft_freqs_side = np.array(freqs_side)
    plt.subplot(311)
    p1 = plt.plot(t, signal, "g")

    # uncomment this part to show the plots using matplotlib
    # plotting the signal
    '''plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    p2 = plt.plot(freqs, FFT, "r")
    # plotting the complete fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.subplot(313)
    p3 = plt.plot(freqs, abs(FHT.real), "b")
    # plotting the positive fft spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()'''

    # append the data in one single array
    # use only the real values of the FHT as Discrete Harley Transform only uses real values
    svmdata.append(abs(FHT.real))
    
    
folder = r"D:\Projects\Thesis\Dataset\GIRL"
for filename in os.listdir(folder):
    fs_rate, signal = wavfile.read(os.path.join(folder,filename))
    print("Frequency sampling", fs_rate)
    signal = np.resize(signal,(130000,2))
    l_audio = len(signal.shape)
    print("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print("secs", secs)
    Ts = 1.0/fs_rate
    print("Timestep between samples Ts", Ts)
    t = np.arange(0, secs, Ts)
    t = np.resize(t,(130000,))
    FFT = abs(scipy.fft.fft(signal))
    FHT = scipy.fft.fft(signal) * np.array([1 + 1j])
    FFT_side = FFT[range(N//2)]
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)]
    fft_freqs_side = np.array(freqs_side)
    '''plt.subplot(311)
    p1 = plt.plot(t, signal, "g")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    p2 = plt.plot(freqs, FFT, "r")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count dbl-sided')
    plt.subplot(313)
    p3 = plt.plot(freqs, abs(FHT.real), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.show()'''

    svmdata.append(abs(FHT.real))

# labels used for prediction
for y in range(0,16):
    label.append('Boy')
for y in range(0,16):
    label.append('Girl')

# set the SVM Classifier and it's parameters (you can adjust the parameters as you wish)
model = svm.SVC(kernel ='linear', C=1)

model.fit(svmdata, label)

decision_tree_pkl_filename = 'boy_vs_girl_svm_dht_model.pkl'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(model, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()
