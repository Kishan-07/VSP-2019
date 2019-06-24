# VSP-2019

Codes for parameter estimation for two different cosmological models are provided. The details have been explained in comments in the beginning of the codes.

Supernova Ia data from three surveys: JLA, SCP and Reiss et al. (2004) have been included in the repository. The matrices 'dl', for different surveys, saved in various .npy files are also provided. If you are using the same parameter limits originally given in the code, the code will automatically load the matrix 'dl' from the corresponding .npy file.

Download these files and change their path in the codes to the path where you saved the corresponding files.

A typical command to run a code would look like this:
```
run non_flat.py S JLA N 1000000
```
The first argument following `S` is the name of the survey whose data you want to use. It could be: 'JLA', 'SCP' or 'Reiss'. The second command following `N` takes the length of the MCMC chain you want the code to output. This must be greater than or equal to 10000. If the length of the chain is not given in the command, code will take it to be 10000 by default.

You can play with the code by changing the variance of the prior distribution, the size of the proposal distribution, the limits of parameters, length of MCMC chain, etc.
