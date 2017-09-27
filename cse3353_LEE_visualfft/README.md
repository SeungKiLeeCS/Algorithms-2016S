Fast Fourier Transform Visualization : Real Time Spectrum Analyzer
=========

Execution
------------

1) Open command line and direct to the folder
2) python visualizefft.py

Functionality
------------

A basic implementation of Spectrum Analyzer. Reads data from the sound card and
displays the time history / spectrum in real time.

Requirements for Execution
------------

pyside : For Python binding with QT
numpy, scipy : Compute Fourier Transform
pyaudio : Audio Processing
pyqtgraph : Real Time Plot (Faster than matplotlib)
