import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")

whisky.head()

whisky.iloc[0:10] #we specify the rows of from 0 to 10

whisky.columns

flavors = whisky.iloc[:, 2:14]

# pearson correlation between flavors
corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)
plt.figure(figsize =(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf")


# pearson correlation between whisky distilleries in terms of the flavor profiles
corr_whisky = pd.DataFrame.corr(flavors.transpose()) #matris transpozu
print(corr_whisky)
plt.figure(figsize =(10,10))
plt.pcolor(corr_whisky, cmap = "gist_rainbow")
plt.axis("tight")
plt.colorbar()
plt.savefig("corr_whisky.pdf")


#Spectral co-clustering

model = SpectralCoclustering(n_clusters = 6, random_state=0) #6 different region
model.fit(corr_whisky)
model.rows_

#Each row in this array identifies a cluster, here ranging from 0 to 5,
#and each column identifies a row in the correlation matrix,
#here ranging from 0 to 85.



#The output tells us how many whiskeys belong to a cluster 0,
#cluster 1, cluster 2, and so on.
#For example, here, 19 whiskeys belong to cluster number 2.
np.sum(model.rows_, axis= 1)

#Observation number 0 belongs to cluster number 5,
#observation number 1 belongs to cluster number 2, and so on.
model.row_labels_

#draw the clusters as groups
#extract group labels from the model
whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)
#reorder the rows in increasing order by group labels
#These are the group labels that we discovered
#using spectral co-clustering.
whisky = whisky.ix[np.argsort(model.row_labels_)]
#Finally, we reset the index of our DataFrame.
whisky = whisky.reset_index(drop = True)

#exercises
data = pd.Series([1,2,3,4])
data = data.ix[[3,0,1,2]]

dataX = pd.Series([1,2,3,4])
dataX = dataX.ix[[3,0,1,2]]
dataX = dataX.reset_index(drop=True)


#We first used the iloc.
#We're taking all of the rows,
#and then we'd like to specify columns from 2 to 14.
#We need to transpose this table as before.
#And to calculate correlations, we use pd.DataFrame.corr.
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())

#Let's then turn this into a NumPy array.
correlations = np.array(correlations)

#make a plot of the original correlation coefficients
#and the rearranged correlation coefficients,
#which we would expect to form clusters.
#use the pcolor function to plot the two correlation matrices.
#Our original correlation matrix was called core underscore whisky,
#and the new order correlation matrix is called correlations.
plt.figure(figsize =(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky, cmap=plt.cm.Spectral)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations, cmap=plt.cm.Spectral)
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("correlations.pdf")



#There are many different kinds of correlations,
#and by default, the function uses what is
#called Pearson correlation which estimates
#linear correlations in the data.
#In other words, if you have measured attributes for two variables,
#let's call them x and y the Pearson correlation coefficient
#between x and y approaches plus 1 as the points in the xy scatterplot approach
#a straight upward line.
#But what is the interpretation of a correlation
#coefficient in this specific context?
#A large positive correlation coefficient indicates
#that the two flavor attributes in question
#tend to either increase or decrease together.
#In other words, if one of them has a high score
#we would expect the other, on average, also to have a high score.
