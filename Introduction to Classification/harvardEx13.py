import numpy as np 
import scipy.stats as ss
import random

def distance(p1, p2):
    """find the distance between point s p1 and p2."""
    return np.sqrt(np.sum(np.power(p2-p1, 2)))

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
            
    return random.choice(winners)  #our function will return only just one winner
                                   #even in the case of a tie
                                   


def majority_vote_short(votes):
    """Return most common element in votes"""
    mode, count = ss.mstats.mode(votes)
    return mode       
   


votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2]
vote_counts = majority_vote(votes)
                   