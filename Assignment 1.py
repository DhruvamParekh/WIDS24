#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install necessary libraries
# Uncomment and run the next line to install libraries if they are not installed.
# pip install numpy librosa scipy matplotlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


# In[14]:


audio_file = "C:\Dhruvam\WiDS\O Mere Dil Ke Chain - Mere Jeevan Saathi 320 Kbps.mp3"  # Provide the path to your audio file
y, sr = librosa.load(audio_file, sr=None)

# Plot the signal
plt.figure(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.8)
plt.title("Waveform of the Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# In[10]:


# Perform Fourier Transform on the signal
Y = np.fft.fft(y)  # Compute the Fast Fourier Transform (FFT)
magnitude = np.abs(Y)  # Compute the magnitude of the FFT
frequencies = np.fft.fftfreq(len(Y), d=1/sr)  # Compute the frequency bins

# Plot the magnitude spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])  # Plot only positive frequencies
plt.title("Frequency Spectrum of the Audio Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# In[26]:


from scipy.signal import butter, filtfilt
import soundfile as sf

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  
    normal_cutoff = cutoff / nyquist 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
cutoff_frequency = 4000
order = 6

b, a = butter_lowpass(cutoff_frequency, sr, order)
filtered_signal = filtfilt(b, a, y)

sf.write("filtered_audio.wav", filtered_signal, sr)
plt.figure(figsize=(12, 6))
librosa.display.waveshow(filtered_signal, sr=sr, alpha=0.8, color='r')
plt.title("Filtered Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# In[5]:


class Neuron:
    def __init__(self, num_inputs, weights):
        self.num_inputs = num_inputs  
        self.weights = weights        
        self.bias = 1.0              

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(total)

# Test the Neuron
weights = np.array([0.2, 0.4, 0.4])  
neuron = Neuron(3, weights)          
inputs = np.array([0.5, 0.3, 0.8])
output = neuron.feedforward(inputs) 
print('Output from the neuron:', output)


# In[7]:


class Neuron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def feedforward(self, inputs):
        self.inputs = inputs
        self.total_input = np.dot(inputs, self.weights) + self.bias
        self.output = self.sigmoid(self.total_input)
        return self.output
    def train(self, inputs, expected_output, epochs=1000):
        for epoch in range(epochs):
            output = self.feedforward(inputs)
            error = expected_output - output
            derror = error * self.sigmoid_derivative(output)
            self.weights += self.learning_rate * derror * inputs
            self.bias += self.learning_rate * derror


# In[ ]:




