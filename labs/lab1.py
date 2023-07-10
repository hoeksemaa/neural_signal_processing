import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
import scipy as sp
import itertools as it
sns.set_style('whitegrid')

Fs = 30000     # sampling rate of the signal in Hz
dt = 1/Fs
gain = .5      # gain of the signal
x = pd.read_csv('~/Desktop/data/nda_ex_1.csv', header=0, names=('Ch1', 'Ch2', 'Ch3', 'Ch4'))

def filterSignal(x, Fs, low, high):
    # Define order of the filter
    order = 2

    # Define the frequency band
    nyquist = 0.5 * Fs
    low = low / nyquist
    high = high / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply the filter
    y = pd.DataFrame()
    for column in x:
        y[column] = signal.filtfilt(b, a, x[column])

    return y

xf = filterSignal(x, Fs, 500, 4000)

plt.figure(figsize=(14, 8))

T = 100000
t = np.arange(0,T) * dt 

for i, col in enumerate(xf):
    plt.subplot(4,2,2*i+1)
    plt.plot(t,x[col][0:T],linewidth=.5)
    plt.ylim((-1000, 1000))
    plt.xlim((0,3))
    plt.ylabel('Voltage')
    
    
    plt.subplot(4,2,2*i+2)
    plt.plot(t,xf[col][0:T],linewidth=.5)
    plt.ylim((-400, 250))
    plt.xlim((0,3))
    plt.ylabel('Voltage')

plt.show()
