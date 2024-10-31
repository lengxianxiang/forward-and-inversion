# forward-and-inversion
A method for full waveform inversion using deep learning.
The experimental environment used for this method is described: python 3.7 and Tensorflow 1.15.

forward
The files required for the forward simulation of the OSPRK method are: model.py, osprk-4-pml.py, snapshot.m.
Using the model.py and osprk-4-pml.py files, you can get the orthogonal wavefield values, and then you can get a snapshot of the wavefield by running the snapshot.m file using matlab.

inversion
The files needed for the inversion are: setup_seam.py, onsprk-4-pml.py, log_cosh.py, NADAM.py.
The setup_seam.py file is mainly the input for the real model and the initial model.
The osprk-4-pml.py file is mainly code for the OSPRK method.
The log_cosh.py file focuses on the loss function and the setup of the inversion parameters.
The NADAM.py file is mainly the code needed for plotting.
