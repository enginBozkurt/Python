import numpy as np
import matplotlib.pyplot as plt
import random

points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]])

p= np.array([2.5, 2])
#x koordinatları için 0.sütundaki tüm satırlar, y koordinatları için  1.sütundaki tüm satırlar
plt.plot(points[:,0], points[:,1], "ro")
#P[0] birinci eleman, P[1] ikinci eleman
plt.plot(p[0], p[1], "bo")
plt.axis([0.5, 3.5, 0.5, 3.5])


def distance(p1, p2):
    """find the distance between point s p1 and p2."""
    return np.sqrt(np.sum(np.power(p2-p1, 2)))



#uzaklıkları tutacak arrray, point dizisindeki satır sayısına eşit sayıda elemanı olacak
distances = np.zeros(points.shape[0])
for i in range(len(distances)):
    #p noktası ile points array indeki noktalar arası uzaklığın hesaplanması
    distances[i] = distance(p, points[i])
    
#sıralanmış dizinin indislerini döner
ind = np.argsort(distances)
distances[ind]  

#en yakın iki nokta
distances[ind[0:2]]  

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

#k nearest neighbors function
def find_nearest_neighbors(p, points, k=5):
    """Find the k nearest neighbors of point p and return their indices"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
    #p noktası ile points array indeki noktalar arası uzaklığın hesaplanması
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]  #0 dan k ya kadar olan tüm indisler
    

def knn_predict(p, points, outcomes, k=5):
    #find k nearest neighbors
    ind = find_nearest_neighbors(p, points, k)
    #predict the class of p based on majority vote
    return majority_vote(outcomes[ind])
    
#class 0 and class 1
outcomes = np.array([0,0,0,0,1,1,1,1,1])

knn_predict(np.array([2.5, 2.7]), points, outcomes, k=2)


