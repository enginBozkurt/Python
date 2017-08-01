import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
#mean=0 , standard deviation=1 is normal distribution for obsv. coming fom class 0
#mean=1 , standard deviation=1 is normal distribution for obsv. coming fom class 1
#generating synthetic data

def generate_synth_data(n=50):
    """Create two sets of points from bivariate normal distributions """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)

n = 20
(points, outcomes) = generate_synth_data(n)
plt.figure()
plt.plot(points[:n,0], points[:n,1], "ro")  #first n rows
plt.plot(points[n:,0], points[n:,1], "bo")  #remaining rows
plt.savefig("bivardata.pdf")



    
