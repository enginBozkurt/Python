import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

birddata = pd.read_csv("bird_tracking.csv")
ix = birddata.bird_name ==  "Eric"

speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], ec='black')    #nümerik olan değerler histograma gönderildi
plt.savefig("hist.pdf")

np.sum(np.isnan(speed))


plt.figure(figsize= (8,4))
ix = birddata.bird_name ==  "Eric"
speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
#In this case, the first bin starts at 0 and the last bin ends at 30.
#Finally, I have normalized the y-axis, meaning
#that an integral over the histogram would be equal to 1.
plt.hist(speed[~ind], bins = np.linspace(0, 30, 20), normed = True, ec='black')
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.savefig("birdSpeed.pdf")



x, y = birddata.longitude[ix], birddata.latitude[ix]
plt.figure(figsize =(7,7))
plt.plot(x, y, ".")

bird_names = pd.unique(birddata.bird_name)

plt.figure(figsize =(7,7))
for bird_name in bird_names:
    ix = birddata.bird_name ==  bird_name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x, y, ".", label= bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc= "lower right")    
plt.savefig("3traj.pdf")    


#plot by using Pandas, you do not need to deal with  NaNs explicitly
birddata.speed_2d.plot(kind= 'hist', range=[0, 30])
plt.xlabel("2D speed")



#USING DATETIME OBJECT
#If we'd like to compute how much time has passed between any two
#observations in our data set, we first have
#to convert the timestamps, now given as strings, to datetime objects.
#Let's first pull out the first timestamp from our data set.
#That's located at row 0, so bird data dot date time.
date_str = birddata.date_time[0] 
date_str[:-3] #with the exception of the last three characters.

datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")        


#empty list
timestamps = []
#For every single row, I extract the date time,
#except I ignore the last three characters
#Once I've converted it to a datetime object
#I append the resulting object to my timestamps list.
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
                  (birddata.date_time.iloc[k][:-3],  "%Y-%m-%d %H:%M:%S"))
    
    
timestamps[0:3]

#The next step for me is to construct a panda series object
#and insert the timestamp from my Python list into that object.
#I can then append the panda series as a new column in my bird data data frame
#On the right-hand side, I first create a panda series object
#from my timestamps list, and I explicitly specify the index
#to match the index of my bird data.
#On the left-hand side I then take my existing bird data.
#I create a new column, which is called timestamp,
#and I assign my panda series object that new timestamp column.
#In this case, we can see that the timestamp
#has been appended, or attached, to our table as the final column.


birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)

birddata.head()

#What I'd like to do next is to create a list that
#captures the amount of time that has elapsed
#since the beginning of data collection.
#To do this, we will need 2 lines of code.
#First I will extract the timestamps for Eric,
#and that object is going to be called times.
#I then create my elapsed time object and I
#construct that as a list comprehension.
#Let's look at the list comprehension in a little more detail.
#Let's first focus on the for loop for time in times.
#What happens is that we're taking the time sequence and going over it
#one object at a time.
#These objects are called time-- that's our loop variable.
#The element that gets appended the list is the following--
#that's given at the beginning of the list comprehension--
#time minus times at 0.
#In other words, for each object in the list, time is going to be different--
#whereas times square bracket 0 marks the beginning of the time measurement,
#in this case, for Eric.


#time stamps for bird Eric
times = birddata.timestamp[birddata.bird_name == "Eric"]

#the list comprehension
elapsed_time =[time - times[0] for time in times]

elapsed_time[1000]

#I'm going to make a plot where on the x-axis,
#we have the number of the observation, and on the y-axis
#we have the amount of time that has elapsed, measured in days

plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time (days)")
plt.savefig("timeplot.pdf")


#calculating daily mean speed


#Our data consists of time stamps which are spaced unevenly.
#This is our time axis here.
#The first point on the left corresponds to our observation number 0.
#This is observation number 1, 2, 3, 4, 5, 6, 7, 8, and so on.
#All time is measured relative to our observation 0.
#If we were to examine the first few points here,
#we would find that they likely correspond to day 0.
#The first day of the experiment.
#We'd like to be collecting the indices for all those observations
#that fall within day zero
#As soon as we hit day 1, we would like to compute the mean velocity
#over those observations that we have just collected.
#We would then like to start collecting, or aggregating,
#our indices for the new day.

#To say that differently, we'd like to be collecting indices corresponding
#to different observations until we hit the next day.
#Our starting point is day zero.
#So I'm going to be using a variable called next day
#and that's going to be equal to 1.
#I also need a way to keep track of all the different indices.
#I'm going to call that inds.
#Let's say that these three observations correspond to day zero.
#These to day 1, day 2, and so on.
#I will be going over all of my observations.
#I will always check.
#Have I hit the next day yet.
#The day for these first three observations is 0,
#so the test will be false.
#As soon as I hit this observation, I know that I've hit the next day.
#As I loop over all of these points, I will
#be building up this list on the fly.
#So that by the time I hit observation number 3,
#I will have indices 0, 1, and 2 in my inds list.
#Once this happens, I do the following:
#I first compute the average velocity over these indices
#for the previous or the current day.
#And I then reset my next day to be equal to plus 1 to what I was before.
#So in this case, I'll be setting that to be equal to 2.
#I will proceed this way until I have no more data left


data = birddata[birddata.bird_name == "Eric"]
times = data.timestamp
elapsed_time =[time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)


next_day = 1
inds = []  # empty list
daily_mean_speed = []
#I'd like to know both the index of any given observation as well as its value.
#I can accomplish this using enumerate
for (i ,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = [] 
        
plt.figure(figsize =(8,6))
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean speed (m/s)")
plt.savefig("dms.pdf")

# projected trajectories on map  using cartopy

# We're first looping over all of our bird names.
# Then we're extracting the rows from our data
# frame to correspond to that particular bird.
# Then we extract the longitude and latitude in variables x and y.
# The final line, the plot line, is what introduces the transformation.

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for name in bird_names:
    ix = birddata['bird_name'] == name
    x, y = birddata.longitude[ix], birddata.latitude[ix] 
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)       
                 
                 
plt.legend(loc="upper left")  
plt.savefig("map.pdf")  

