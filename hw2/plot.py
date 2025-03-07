import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import wave

def no1_pic_data():
    # Read the .wav file
    rate, data = wavfile.read('hw2/star_noisy.wav')
    # Get the first 16000 samples
    data = data[:16000]

    data = data / (np.max(np.abs(data)))
    data = data - np.mean(data)

    plt.subplot(3, 1, 1)
    plt.title('Waveform of star_noisy.wav')
    plt.xlabel('Sample Index')
    plt.xlim(0, 16000)
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    plt.yticks([-1, -0.5,0, 0.5, 1])
    for i in range(0, 16000, 2000):
        if i == 6000:
            continue
        plt.axvline(x=i, color='black', linestyle=':', linewidth=0.5)
    plt.axhline(y=0.5, color='black', linestyle=':', linewidth=0.5)
    plt.axhline(y=-0.5, color='black', linestyle=':', linewidth=0.5)
    plt.plot(data, lw=0.5,color='b')
    plt.axvline(x=6000, color='r', linestyle='-', linewidth=0.5)
    plt.axvline(x=6256, color='r', linestyle='-', linewidth=0.5)
    return data

def no2_pic_data():
    # Read the .wav file
    data = no1_pic_data()
    data = data[6000:6256]
    plt.subplot(3, 1, 2)
    plt.plot(data, lw=0.5,color='b')
    plt.yticks([0.1, 0, -0.1, -0.2])
    plt.grid(linestyle=":")
    plt.xlim(0, 256)
    conefs = np.polyfit(range(256), data, 3)
    poly = np.polyval(conefs, range(256))
    plt.plot(poly, lw=0.5,color='r')
    return data, poly

def no3_pic_data():
    # Read the .wav file
    data, poly = no2_pic_data()
    data = data - poly
    plt.subplot(3, 1, 3)
    plt.plot(data, lw=0.5,color='b')
    plt.yticks([0.15, 0.1, 0.05,0, -0.05])
    plt.grid(linestyle=":")
    plt.xlim(0, 256)
    

if __name__ == '__main__':
    # Plot the waveform
    data = no1_pic_data() 
    no2_pic_data()
    no3_pic_data()
    plt.show() 