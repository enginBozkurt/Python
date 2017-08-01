import numpy as np
import random
import scipy.stats as ss

def make_prediciton_grid(predictors, outcomes, limits, h, k):
    """Classify each point on the prediction grid"""
    (xmin, xmax, ymin, ymax) = limits
    xs = np.arange(xmin, xmax, h)
    ys = np.arange(ymin, ymax, h)
    xx, yy = np.meshgrid(xs, ys)  #return coordinate matrices from coordinate vectors
    #generate our classifiers prediction corresponding to every point of the meshgrid
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k) #y değerleri satır, x değereleri sütun
    return (xx, yy,  prediction_grid)
            
#Meshgrid takes in two or more coordinate vectors,
#say one vector containing the x values of interest and the other containing
#the y values of interest.
#It returns matrices, the first containing the x values
#for each grid point and the second containing the y values for each grid point.
                     
def knn_predict(p, points, outcomes, k=5):
    #find k nearest neighbors
    ind = find_nearest_neighbors(p, points, k)
    #predict the class of p based on majority vote
    return majority_vote(outcomes[ind])

def find_nearest_neighbors(p, points, k=5):
    """Find the k nearest neighbors of point p and return their indices"""
    distances = np.zeros(points.shape[0]) #shape[0] points array in satır sayısı
    for i in range(len(distances)):
    #p noktası ile points array indeki noktalar arası uzaklığın hesaplanması
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]    

def majority_vote(votes):
    """Return most common element in votes"""
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
                       
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
            
    return random.choice(winners)

def distance(p1, p2):
    """find the distance between point s p1 and p2."""
    return np.sqrt(np.sum(np.power(p2-p1, 2)))


def generate_synth_data(n=50):
    """Create two sets of points from bivariate normal distributions """
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)


 
#plotting grid function  
def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)                                  

# try samples

(predictors, outcomes) = generate_synth_data()

k=5; filename= "knn_synth_ 5.pdf"; limits = (-3,4,-3,4); h = 0.1
(xx, yy, prediction_grid) =    make_prediciton_grid(predictors, outcomes, limits, h, k)   
plot_prediction_grid (xx, yy, prediction_grid, filename)
                                           
k=50; filename= "knn_synth_ 50.pdf"; limits = (-3,4,-3,4); h = 0.1
(xx, yy, prediction_grid) =    make_prediciton_grid(predictors, outcomes, limits, h, k)   
plot_prediction_grid (xx, yy, prediction_grid, filename)


# for iris flowers dataset(3 different types)
from sklearn import datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()      #all datasets
#all rows but only two columns
predictors = iris.data[:, 0:2]
outcomes = iris.target
plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")
plt.savefig("iris.pdf")


k=5; filename= "iris_grid.pdf"; limits = (4,8,1.5,4.5); h = 0.1
(xx, yy, prediction_grid) =    make_prediciton_grid(predictors, outcomes, limits, h, k)   
plot_prediction_grid (xx, yy, prediction_grid, filename)


#kNN algorithm using SciKit library
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)



#We can now try out different values for k.
#If you use a small value you'll see that the boundary between the colors,
#the so-called decision boundary, is more smooth the larger the value of k.
#This means that k controls the smoothness of the fit.




#It turns out that using a value for k that's too large or too small
#is not optimal.
#A phenomenon that is known as the bias-variance tradeoff.
#This suggests that some intermediate values of k might be best.
#We will not talk more about it here,
#but for this application, using k equal to 5 is a reasonable choice.

