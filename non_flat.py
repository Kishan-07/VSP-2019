# This code evaluates the comological parameters omega_m, omega_l and h using Supernova Ia data which contains pairs of mu and z. Metropolis algorithm is employed. Errors are assumed to Gaussian, i.e. likelihood is assumed to be a multivariate Guassian with certain covariance matrix. The cosmological model assumed is non-flat LCDM with non-evolving dark energy. Refer to MCMC and HMC exercise by Alan Heavens

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.signal import savgol_filter

# Prior, which we take to be Gaussain. You can play with the variance of the Gaussian.
def prior(omega_m, omega_l, h):
    return np.exp(-(omega_m-0.3)**2/3200 - (omega_l-0.7)**2/3200 - (h-0.7)**2/3200)

# Evaluates luminosity distance with h = 1. Since we are dealing with non-flat universe, we have three cases: open, cloased, flat
def dL(z, omega_m, omega_l):
    N = len(z)
    
    # If the quantity "omega_m*(1+x)**3 + omega_l + (1-omega_m-omega_l)*(1+x)**2" is negative, it would create problem when we take root while evaluating dL. The MCMC chain will never go there, and so not create any problem.
    x = np.linspace(0, max(z), 101)
    if np.sum(omega_m*(1+x)**3 + omega_l + (1-omega_m-omega_l)*(1+x)**2 <= 0) > 0:
        return 0
    
    # 3 cases
    if omega_m + omega_l == 1:
        f = lambda x: (omega_m*(1+x)**3 + omega_l)**(-0.5)
        int_val = np.zeros(N)
        for i in range(N):
            int_val[i] = integrate.quad(f, 0, z[i])[0]
        ret_val = 3000*(1+z)*int_val
    elif omega_m + omega_l < 1:
        f = lambda x: (omega_m*(1+x)**3 + omega_l + (1-omega_m-omega_l)*(1+x)**2)**(-0.5)
        int_val = np.zeros(N)
        for i in range(N):
            int_val[i] = integrate.quad(f, 0, z[i])[0]
        r = (np.abs(1-omega_m-omega_l))**0.5*int_val
        ret_val = 3000*(1+z)*np.sinh(r)/(np.abs(1-omega_m-omega_l))**0.5
    else:
        f = lambda x: (omega_m*(1+x)**3 + omega_l + (1-omega_m-omega_l)*(1+x)**2)**(-0.5)
        int_val = np.zeros(N)
        for i in range(N):
            int_val[i] = integrate.quad(f, 0, z[i])[0]
        r = (np.abs(1-omega_m-omega_l))**0.5*int_val
        ret_val = 3000*(1+z)*np.sin(r)/(np.abs(1-omega_m-omega_l))**0.5

    return ret_val

# Evaluating the luminosity distance for a grid of omega_m and omega_l. We would later just interpolate to find the values. Returns a 3-dimensional matrix.
def dl_matrix(z):
    global omega_m_lim, omega_l_lim
    
    dl = np.zeros([len(z), int(100*(omega_m_lim[1] - omega_m_lim[0])) + 1, int(100*(omega_l_lim[1] - omega_l_lim[0])) + 1])
    for i in range(int(100*(omega_m_lim[1] - omega_m_lim[0])) + 1):
        for j in range(int(100*(omega_l_lim[1] - omega_l_lim[0])) + 1):
            dl[:, i, j] = dL(z, 0.01*i + omega_m_lim[0], 0.01*j + omega_l_lim[0])

    return dl

# Function to carry out 2D interpolation
def interpol_2d(omega_m, omega_l):
    global dl, omega_m_lim, omega_l_lim

    # index of one of the lower left corner of box containing (omega_m, omega_l)
    i = int(np.floor(100*(omega_m - omega_m_lim[0])))
    j = int(np.floor(100*(omega_l - omega_l_lim[0])))

    return (dl[:, i, j]*(0.01*i + omega_m_lim[0] + 0.01 - omega_m)*(0.01*j + omega_l_lim[0] + 0.01 - omega_l) + dl[:, i+1, j]*(omega_m - 0.01*i - omega_m_lim[0])*(0.01*j + omega_l_lim[0] + 0.01 - omega_l) + dl[:, i, j+1]*(0.01*i + omega_m_lim[0] + 0.01 - omega_m)*(omega_l - 0.01*j - omega_l_lim[0]) + dl[:, i+1, j+1]*(omega_m - 0.01*i - omega_m_lim[0])*(omega_l - 0.01*j - omega_l_lim[0]))*10000

# Evaluate the difference mu = m - M theoretically for given z and parameters
def mu_th(z, omega_m, omega_l, h):
    return 25 + 5*np.log10(interpol_2d(omega_m, omega_l)/h)

# Evaluates the exponent in Gaussian approximation of the likelihood function as defined in the exercise.
def likelihood_exponent(omega_m, omega_l, h):
    global z, mu, curvmat
        
    mu_diff = mu - mu_th(z, omega_m, omega_l, h)
    ret_val = -0.5*np.matmul(mu_diff, np.matmul(curvmat, mu_diff))
    
    return ret_val

# Generates new parameter values in the vicinity of old parameter values from a square top-hat proposal distribution with certain size.
def gen_new_sample(omega_m, omega_l, h):
    global size
    
    domega_m = size*np.random.random() - size/2.0
    domega_l = size*np.random.random() - size/2.0
    dh = size*np.random.random() - size/2.0
    
    return omega_m + domega_m, omega_l + domega_l, h + dh

# Using the Metropolis algorithm, decide whether to accept the new point.
def move(omega_m, omega_l, h, likelihood_exp):
    global acceptance, omega_m_lim, omega_l_lim, h_lim
    
    omega_m_new, omega_l_new, h_new = gen_new_sample(omega_m, omega_l, h)
    
    # If generated values of parameters are outside the bound, likilhood_exp  = -infinity so that post_ratio is 0 and the point will not be accepted.
    if omega_m_new < omega_m_lim[0] or omega_m_new > omega_m_lim[1] or omega_l_new < omega_l_lim[0] or omega_l_new > omega_l_lim[1] or h_new < h_lim[0] or h_new > h_lim[1]:
        likelihood_exp_new = -np.inf
    else:
        likelihood_exp_new = likelihood_exponent(omega_m_new, omega_l_new, h_new)
    
    post_ratio = np.exp(likelihood_exp_new - likelihood_exp)*prior(omega_m_new, omega_l_new, h_new)/prior(omega_m, omega_l, h)
    rand = np.random.random()
    
    if rand < post_ratio:
        acceptance = acceptance + 1
        ret_val = omega_m_new, omega_l_new, h_new, likelihood_exp_new
    else:
        ret_val = omega_m, omega_l, h, likelihood_exp
    
    return ret_val

# Plot the probability distribution of single parameter chain by binning
def plot_1d(chain, ax):
    hist = np.histogram(chain, bins=100)

    par_val = hist[1]
    length = len(par_val)
    x = (par_val[0:length-1] + par_val[1:length])/2.0
    
    # Smoothen the step probability distribution using Savitzky-Golay filter
    prob = savgol_filter(hist[0], window_length = 51, polyorder = 6)

    # Normalise
    prob = 1.0*prob/integrate.trapz(prob, x)

    ax.plot(x, prob, '#004C99')
    
    return

# Plot the 68.27%, 95.45% and 99.73% confidence intervals for a given pair of parameter arrays
def plot_2d(chain1, chain2, ax):
    curv_mat = np.linalg.inv(np.cov([chain1, chain2]))
    
    chain1_mean = np.mean(chain1)
    chain2_mean = np.mean(chain2)
    
    chain1_std = np.std(chain1)
    chain2_std = np.std(chain2)

    x = np.linspace(chain1_mean - 5*chain1_std, chain1_mean + 5*chain1_std, 1001)
    y = np.linspace(chain2_mean - 5*chain2_std, chain2_mean + 5*chain2_std, 1001)
    x, y = np.meshgrid(x, y)
    
    z = ((x-chain1_mean)**2)*curv_mat[0, 0] + 2*(x-chain1_mean)*(y-chain2_mean)*curv_mat[0, 1] + ((y-chain2_mean)**2)*curv_mat[1, 1]
    
    ax.contour(x, y, z, levels=[2.295815, 6.180086, 11.829007], colors = ['#003366', '#004C99', '#0066CC'])
    ax.contourf(x, y, z, levels=[0, 2.295815, 6.180086, 11.829007], colors = ['#003366', '#0066CC', '#3399FF'])
    
    return

# Creates a typical corner plot. 'chains' contain array of N chains (here 3 chains) and 'names' contain the strings used to label the axes of plots.
def corner_plot(chains, names):
    N = chains.shape[0]
    
    means = chains.mean(axis = 1)
    std = chains.std(axis = 1)
    
    # Limits for axes in the plot
    ax_lims = np.transpose(np.array([means - 4*std, means + 4*std]))
    
    # We don't want matter density to be negative
    if ax_lims[0, 0] < 0:
        ax_lims[0, 0] = 0.0
    
    fig = plt.figure(figsize = [7, 7])
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # This loop produces the single parameter probability distribtuions
    for i in range(N):
        ax = plt.subplot(N, N, (N + 1)*i + 1)
        plot_1d(chains[i], ax)
        ax.set_xlim([ax_lims[i, 0], ax_lims[i, 1]])
        ax.locator_params(axis = 'x', nbins=3)
        ax.locator_params(axis = 'y', nbins=8)
        
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        
        if i != N-1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax.set_xlabel(names[i])

    # This loop produces the two-parameter confidence intevals
    for i in range(N):
        for j in range(N):
            if j >= i:
                continue
            
            ax = plt.subplot(N, N, N*i + j + 1)
            plot_2d(chains[j], chains[i], ax)
            
            ax.set_xlim([ax_lims[j, 0], ax_lims[j, 1]])
            ax.set_ylim([ax_lims[i, 0], ax_lims[i, 1]])
            ax.locator_params(nbins=3)
            
            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            
            if i == N-1:
                ax.set_xlabel(names[j])
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
    
    return



###################################################################################################
########################################     Main Code     ########################################
###################################################################################################

# Default value of length of MCMC chain if length is not provided in command line argument
length = 10000

# Command line argument. After 'S', the user has to give the name of survey, either 'JLA', 'SCP' or 'Reiss'. Corresponding data files have been provided at https://github.com/Kishan-07/VSP-2019
# After 'N', the length of the chain must be specified. If length is not specified, default value of 10000 is taken from which first 5000 will be discarded in burn-in.
sysarg = sys.argv
if len(sysarg) == 3:
    if sysarg[1] == 'S':
        survey = sysarg[2]
    else:
        print 'Error: Command line argument must atleast have the survey name after \'S\', and optionally the length of chain after \'N\''
        sys.exit()
elif len(sysarg) == 5:
    if sysarg[1] == 'S' and sysarg[3] == 'N':
        survey = sysarg[2]
        length = int(sysarg[4])
    elif sysarg[3] == 'S' and sysarg[1] == 'N':
        survey = sysarg[4]
        length = int(sysarg[2])
    else:
        print 'Error: Command line argument must atleast have the survey name after \'S\', and optionally the length of chain after \'N\''
        sys.exit()
else:
    print 'Error: Command line argument must atleast have the survey name after \'S\', and optionally the length of chain after \'N\''
    sys.exit()

if length < 10000:
    print 'Error: \'N\' must be greater than or equal to 10000'
    sys.exit()

if survey == 'JLA':
    # Importing the data and evaluating curvature matrix - the inverse of covariance matrix
    z, mu = np.loadtxt('/Users/kishan/Desktop/VSP/MCMC codes/JLA data/jla_data.txt', skiprows=1).T
    covmat = np.loadtxt('/Users/kishan/Desktop/VSP/MCMC codes/JLA data/jla_covmatrix.txt', skiprows=1)
    covmat = covmat.reshape([31, 31])
    curvmat = np.linalg.inv(covmat)
    
    # Limits for the parameters
    omega_m_lim = np.array([0.0, 0.8])  # By default (0.0, 0.8)
    omega_l_lim = np.array([0.0, 1.2])  # By default (0.0, 1.2)
    h_lim = np.array([0.65, 0.75])      # By default (0.65, 0.75)
    
    # Load the luminosity distance for different values of parameters and z, already calculated beforehand. Use this only if you don't change the limits of parameters given above, else create a new 'dl' matrix.
    if omega_m_lim[0] == 0.0 and omega_m_lim[1] == 0.8 and omega_l_lim[0] == 0.0 and omega_l_lim[1] == 1.2:
        dl = np.load('/Users/kishan/Desktop/VSP/MCMC codes/JLA data/JLA_dl_non_flat.npy')
    else:
        dl = dl_matrix(z)
    
    # Size of top-hat proposal distribution
    size = 0.07

elif survey == 'SCP':
    # Importing the data and evaluating curvature matrix - the inverse of covariance matrix
    z, mu, err_mu = np.loadtxt('/Users/kishan/Desktop/VSP/chi_sq_codes/SCP/SCP_data.tsv', skiprows=1, usecols=(1, 2, 3)).T
    covmat = np.diag(err_mu**2)
    curvmat = np.linalg.inv(covmat)
    
    # Limits for the parameters
    omega_m_lim = np.array([0.0, 0.8])      # By default (0.0, 0.8)
    omega_l_lim = np.array([0.0, 1.3])      # By default (0.0, 1.3)
    h_lim = np.array([0.65, 0.75])          # By default (0.65, 0.75)
    
    # Load the luminosity distance for different values of parameters and z, already calculated beforehand. Use this only if you don't change the limits of parameters given above, else create a new 'dl' matrix.
    if omega_m_lim[0] == 0.0 and omega_m_lim[1] == 0.8 and omega_l_lim[0] == 0.0 and omega_l_lim[1] == 1.3:
        dl = np.load('/Users/kishan/Desktop/VSP/MCMC codes/SCP data/SCP_dl_non_flat.npy')
    else:
        dl = dl_matrix(z)
    
    # Size of top-hat proposal distribution
    size = 0.03

elif survey == 'Reiss':
    # Importing the data and evaluating curvature matrix - the inverse of covariance matrix
    z, mu, err_mu = np.loadtxt('/Users/kishan/Desktop/VSP/MCMC codes/Reiss (2004)/Reiss_data.tsv', skiprows=1, usecols=(1, 2, 3)).T
    covmat = np.diag(err_mu**2)
    curvmat = np.linalg.inv(covmat)
    
    # Limits for the parameters
    omega_m_lim = np.array([0.0, 1.0])      # By default (0.0, 0.8)
    omega_l_lim = np.array([0.0, 1.7])      # By default (0.0, 1.3)
    h_lim = np.array([0.60, 0.70])          # By default (0.65, 0.75)
    
    # Load the luminosity distance for different values of parameters and z, already calculated beforehand. Use this only if you don't change the limits of parameters given above, else create a new 'dl' matrix.
    if omega_m_lim[0] == 0.0 and omega_m_lim[1] == 1.0 and omega_l_lim[0] == 0.0 and omega_l_lim[1] == 1.7:
        dl = np.load('/Users/kishan/Desktop/VSP/MCMC codes/Reiss (2004)/Reiss_dl_non_flat.npy')
    else:
        dl = dl_matrix(z)
    
    # Size of top-hat proposal distribution
    size = 0.07

else:
    print 'Error: Survey does not exist in the database'
    sys.exit()


# Markov chains for the parameters
omega_m = np.zeros(length)
omega_l = np.zeros(length)
h = np.zeros(length)

# Evaluating the chain
omega_m[0] = (omega_m_lim[1] - omega_m_lim[0])*np.random.random() + omega_m_lim[0]
omega_l[0] = (1 - omega_l_lim[0])*np.random.random() + omega_l_lim[0]
h[0] = (h_lim[1] - h_lim[0])*np.random.random() + h_lim[0]
likelihood_exp = likelihood_exponent(omega_m[0], omega_l[0], h[0])
acceptance = 0.0

for i in range(1, length):
    omega_m[i], omega_l[i], h[i], likelihood_exp = move(omega_m[i-1], omega_l[i-1], h[i-1], likelihood_exp)

# Burn-in
omega_m = omega_m[5000:length]
omega_l = omega_l[5000:length]
h = h[5000:length]

# Output values and plot
print 'omega_m = ', np.mean(omega_m)
print 'omega_l = ', np.mean(omega_l)
print 'h = ', np.mean(h)
print 'covariance matrix = \n', np.cov([omega_m, omega_l, h])
print 'acceptance rate = ', acceptance/length

corner_plot(np.array([omega_m, omega_l, h]), [r'$\Omega_m$', r'$\Omega_{\Lambda}$', r'$h$'])
plt.show()
