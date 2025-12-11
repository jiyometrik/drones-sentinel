# drones-sentinel

A suite of tools, accompanied by a range of classification pipelines, for drone
detection via RF signals.

All working code can be found in `src`. Run `main.py` to initiate the training pipeline.
Code that was used for trialing is in `sandbox`. 

## notes to devs
- `douglas` directory doesn't run; this is just for reference

## todolist
- calculate welch PSD and STFT (short-time fourier transform)
- split the PSD/STFT by second, for training
- save them to the appropriate directories 
- come up with CNN architecture that actually works