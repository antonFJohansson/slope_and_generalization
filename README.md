# Slope and generalization properties of neural networks

This repo contains the code to reproduce the results in the paper "Slope and generalization properties of neural networks".

The code to replicate the experiments can be found in experiment_folder.

The experiments can be run as follows:
* Set the hyperparameters and model architectures to investigate in experiment_code/allParams/largeParams.txt
* Run allHyperPara.py to create txt-files with the desired information.
* Run main.py --param DESIRED_PARAMETER to train the network architectures and meaure the slope
* The local experiment can be found in experiment_code/localMeasure.py

The visualization code to obtain the figures in the paper can be found in visualization_code.

If there are any questions, contact <johaant@chalmers.se>

