This folder contains the data for a pixel array with finite response time, realistic material and various pixel sizes. The setup is a 100 keV beam, irradiating the whole array homogeneously. The histogram data contains the registered events of one pixel.

The filename designates the incident I_0 fluence rate in photons/mm^2/s. For example, the histogram `I0_rate_1E7.*` is for a fluence rate of 1E7 photons/mm^2/s.

The data is stored in three different and equivalent formats: Numpy arrays, Matlab arrays, and plain text files. Please note I haven't tested the Matlab files.
The first column of the data denotes the centre of the histogram bins in MeV, while the second column denotes the number of registered counts for that bin.
Please note: In case an event exceeds the limits of the histogram, then it is assigned the highest available energy bin.