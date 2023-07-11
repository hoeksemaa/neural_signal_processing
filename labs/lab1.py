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

#plt.figure(figsize=(14, 8))

T = 100000
t = np.arange(0,T) * dt 
"""
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
"""
def robust_std_dev(column):
    median_val = np.median(column)
    mad = np.median(np.abs(column - median_val) / 0.6745 )
    return mad

def detectSpikes(x,Fs):
# Detect spikes
# s, t = detectSpikes(x,Fs) detects spikes in x, where Fs the sampling
#   rate (in Hz). The outputs s and t are column vectors of spike times in
#   samples and ms, respectively. By convention the time of the zeroth
#   sample is 0 ms.
    print(x.shape)
    #for column in x:
    #    print(robust_std_dev(column))
    for i in range(x.shape[1]):
        column = x[:, i]
        print(robust_std_dev(column))

    return (s, t)

T = xf.shape[0]
s, t = detectSpikes(xf.values,Fs)

plt.figure(figsize=(7, 8))

tt = np.arange(0,T) * dt

for i, col in enumerate(xf):
    plt.subplot(4,1,i+1)
    plt.plot(tt,xf[col],linewidth=.5)
    plt.plot(tt[s],xf[col][s],'r.')
    plt.ylim((-400, 400))
    plt.xlim((0.025,0.075))
    plt.ylabel('Voltage')

plt.show()
