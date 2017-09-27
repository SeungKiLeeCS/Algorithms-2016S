#!/usr/bin/env pythonw

#modules imported for ploting, FFT, filtering, and future for // and / operations
from __future__ import division
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from scipy.signal import filtfilt
from numpy import nonzero, diff
import pyqtgraph as pg
from recorder import SoundCardDataSource

# CODE FOR FFT

# From FFT function implementation of Numpy
# The function takes in n, and integer value for length of window and d, a scalar initalized to 1.0 for sample spacing.
# Returns sample frequencies for DFT (Later used with rfft, one dimensional DFT and irfft, inverse of rfft) in float array
# Array length = n//2+1, and the floats are sample frequencies
# In relation to Fourier Series, the array f is composed of :
# f(even) = [0, 1, ..., n/2-1, n/2] / (d*n)
# f(odd) = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)
def float_array_rfft(n, d=1.0):
    #set up error parameter
    if not isinstance(n, int):
        raise ValueError("n is not an integer")
    value = 1.0/(n*d)
    N = n//2 + 1
    results = np.arange(0, N, dtype=int)
    return results * value

def fft_slices(x):
    Nslices, Npts = x.shape
    window = np.hanning(Npts)
    # Calculate FFT using numpy fft.rfft
    fx = np.fft.rfft(window[np.newaxis, :] * x, axis=1)
    # Convert to normalised Power Spectral Density
    Pxx = abs(fx)**2 / (np.abs(window)**2).sum()
    # Scale for one-sided
    Pxx[:, 1:-1] *= 2
    # Scale by frequency to get a result in (dB/Hz)
    return Pxx ** 0.5

# Buffer function with functionality of slice
def fft_buffer(x):
    window = np.hanning(x.shape[0])
    fx = np.fft.rfft(window * x)
    Pxx = abs(fx)**2 / (np.abs(window)**2).sum()
    Pxx[1:-1] *= 2
    return Pxx ** 0.5

# Find the peaks
def find_peaks(Pxx):
    # filter parameters
    b, a = [0.01], [1, -0.99]
    Pxx_smooth = filtfilt(b, a, abs(Pxx))
    peakedness = abs(Pxx) / Pxx_smooth

    # find peaky regions which are separated by more than 10 samples
    peaky_regions = nonzero(peakedness > 1)[0]
    edge_indices = nonzero(diff(peaky_regions) > 10)[0]
    edges = [0] + [(peaky_regions[i] + 5) for i in edge_indices]
    if len(edges) < 2:
        edges += [len(Pxx) - 1]
    peaks = []
    for i in range(len(edges) - 1):
        j, k = edges[i], edges[i+1]
        peaks.append(j + np.argmax(peakedness[j:k]))
    return peaks

# GUI PART
#Class for plotting the results of FFT computations
class GUIVisualize(pg.GraphicsWindow):
    def __init__(self, recorder):
        super(GUIVisualize, self).__init__(title="CSE 3353 Real Time FFT Visulaization")
        self.recorder = recorder
        self.paused = False
        self.logScale = False
        self.showPeaks = False
        self.downsample = True

        # Setup plots and Colors!
        self.p1 = self.addPlot()
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle("Time Domain")
        self.ts = self.p1.plot(pen='r')
        self.nextRow()
        self.p2 = self.addPlot()
        self.p2.setLabel('bottom', 'Frequency', 'Hz')
        self.p2.setTitle("Frequency Domain")
        self.spec = self.p2.plot(pen=(0, 250, 0), brush=(0,250,0), fillLevel=-100)

        # Lines for marking peaks
        self.peakMarkers = []

        # Data ranges
        self.resetRanges()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.recorder.chunk_size / self.recorder.fs)
        # For Testing
        print "Updating graphs every %.1f ms" % interval_ms
        self.timer.start(interval_ms)

    # Set Plotting range
    def resetRanges(self):
        self.timeValues = self.recorder.timeValues
        self.freqValues = float_array_rfft(len(self.timeValues), 1./self.recorder.fs)
        self.p1.setRange(xRange=(0, self.timeValues[-1]), yRange=(-1, 1))
        if not self.logScale:
            self.p2.setRange(xRange=(0, self.freqValues[-1]), yRange=(0, 5))
            self.spec.setData(fillLevel=0)
            self.p2.setLabel('left', 'PSD', '1 / Hz')

    def plotPeaks(self, Pxx):
        # find peaks bigger than some thresholds
        peaks = [p for p in find_peaks(Pxx) if Pxx[p] > 0.3]

        if self.logScale:
            Pxx = 20*np.log10(Pxx)

        # Label peaks
        old = self.peakMarkers
        self.peakMarkers = []
        for p in peaks:
            if old:
                t = old.pop()
            else:
                t = pg.TextItem(color=(0, 50, 0, 50))
                self.p2.addItem(t)
            self.peakMarkers.append(t)
            t.setText("%.1f Hz" % self.freqValues[p])
            t.setPos(self.freqValues[p], Pxx[p])
        for t in old:
            self.p2.removeItem(t)
            del t

    def update(self):
        if self.paused:
            return
        data = self.recorder.get_buffer()
        weighting = np.exp(self.timeValues / self.timeValues[-1])
        Pxx = fft_buffer(weighting * data[:, 0])

        if self.downsample:
            downsample_args = dict(autoDownsample=False, downsampleMethod='subsample', downsample=10)
        else:
            downsample_args = dict(autoDownsample=True)

        self.ts.setData(x=self.timeValues, y=data[:, 0], **downsample_args)
        self.spec.setData(x=self.freqValues, y=(20*np.log10(Pxx) if self.logScale else Pxx))

        if self.showPeaks:
            self.plotPeaks(Pxx)
# Setup plots
app = QtGui.QApplication([])
# Set the sampling rate
FS = 44100
# Setup recorder
recorder = SoundCardDataSource(num_chunks=3, sampling_rate=FS, chunk_size=4*1024)
win = GUIVisualize(recorder)

## Start Qt event loop
if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
