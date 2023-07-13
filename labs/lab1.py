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

print("filtering signal")

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

def check_adjacent_values(column):
    # Initialize a list to store the boolean results
    results = []

    # Iterate over the column starting from the second element and ending at the second-to-last element
    for i in range(1, len(column) - 1):
        # Check the condition for the current element
        condition = column[i] <= column[i-1] and column[i] <= column[i+1]
        
        # Append the result to the list
        if condition:
            results.append(i)

    # Return the list of boolean results
    return results

def detectSpikes(x,Fs):
# Detect spikes
# s, t = detectSpikes(x,Fs) detects spikes in x, where Fs the sampling
#   rate (in Hz). The outputs s and t are column vectors of spike times in
#   samples and ms, respectively. By convention the time of the zeroth
#   sample is 0 ms.

    s = []
    t = []
    dt = 1/Fs

    for i in range(x.shape[1]):
        column = x[:, i]
        std_dev = robust_std_dev(column)
        threshold = -std_dev * 3.5
        #print("checking lowest points")
        lowest_points = check_adjacent_values(column)
        #print(lowest_points)
        #print("evaluating threshold")
        indices = np.where((column < threshold) & np.isin(np.arange(len(column)), lowest_points))[0].tolist()
        s.append(indices)
        t.append([i * dt for i in indices])

    return (s, t)

print("detecting spikes")

T = xf.shape[0]
s, t = detectSpikes(xf.values,Fs)

#plt.figure(figsize=(7, 8))

tt = np.arange(0,T) * dt
"""
for i, col in enumerate(xf):
    plt.subplot(4,1,i+1)
    plt.plot(tt,xf[col],linewidth=.5)
    plt.plot(tt[s[i]],xf[col][s[i]],'r.')
    plt.ylim((-400, 400))
    plt.xlim((0.025,0.075))
    plt.ylabel('Voltage')

plt.show()
"""

def extractWaveforms(x, s, dt):
# Extract spike waveforms.
#   w = extractWaveforms(x, s) extracts the waveforms at times s (given in
#   samples) from the filtered signal x using a fixed window around the
#   times of the spikes. The return value w is a 3d array of size
#   length(window) x #spikes x #channels.
    w = []

    for row in range(len(s)):
        waveforms = [x[index-(round(10 / dt)): index+(round(20 / dt))] for index in s[row]]
        w[row] = waveforms
    return w

print("extracting waveforms")

w = extractWaveforms(xf.values,s, dt)

t = np.arange(-10,20) * dt * 1000

plt.figure(figsize=(11, 8))

for i, col in enumerate(xf):
    plt.subplot(2,2,i+1)
    plt.plot(t,w[i][:,1:100],'k', linewidth=1)
    plt.ylim((-500, 250))
    plt.xlim((-0.33,0.66))
    plt.ylabel('Voltage')

plt.show()

"""
idx = np.argsort(np.min(np.min(w,axis=2),axis=0))


t = np.arange(-10,20) * dt * 1000

plt.figure(figsize=(11, 8))
for i, col in enumerate(xf):
    plt.subplot(2,2,i+1)
    plt.plot(t,w[:,idx[0:100],i],'k', linewidth=1)
    plt.ylim((-1000, 500))
    plt.xlim((-0.33,0.66))
    plt.ylabel('Voltage')

plt.show()
"""
