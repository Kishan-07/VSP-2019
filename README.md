# VSP-2019

Supernova Ia data from three surveys: JLA, SCP and Reiss et al. (2004) have been included in the repository. Download them and change the path to these files in the code to the path where you saved the corresponding data files.

Code for parameter estimation for two different cosmological models have been included. The details have been explained in comments in the beginning of the code itself.

A typical command to run the code would look like this:
```
run non_flat.py S JLA N 1000000
```
The first argument following `S` is the name of the survey whose data you want to use. It could be: 'JLA', 'SCP' or 'Reiss'. The second command following `N` takes the length of the MCMC chain you want the code to output. This must be greater than or equal to 10000. If the length of the chain is not given in the command, code will take it to be 10000 by default.

You can play with the code by changing the variance of the prior distribution and the size of the proposal distribution.
