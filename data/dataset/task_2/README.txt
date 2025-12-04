This folder contains the data for a detector with finite response time, hence causing pile-ups. The setup is a 140 kVp X-ray source beam passing through 33 mm of water or fat, and the spectrum of the parallel X-rays incident on a single detector pixel is registered. 

You find two folders, one for the water and one for the fat phantom. In both folders, you will find the histograms for four rates, that is, the rate of photons after the X-ray source and before the phantom. The rate of photons refers here to the total emitted X-ray beam.

The latter part of the filename designates the rate. For example, the histogram `rate_1E7.*` is for an rate of 1E7 photons/second. Both the X-ray beam and the detector have a cross section of 2*pi*(5mm)^2.

The data is stored in three different and equivalent formats: Numpy arrays, Matlab arrays, and plain text files. Please note I haven't tested the Matlab files.
The first column of the data denotes the centre of the histogram bins in MeV, while the second column denotes the number of registered counts for that bin.
Please note: In case an event exceeds the limits of the histogram, then it is assigned the highest available energy bin.