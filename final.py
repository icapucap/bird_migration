# Python program that predicts the home base of a given species of bird by usinsg the migration data of birds in the species

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import mpl_toolkits
os.environ["PROJ_LIB"] = "/home/shidhu/anaconda3/share/proj"
from mpl_toolkits.basemap import Basemap
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
from mpl_toolkits.mplot3d import Axes3D

# All files opened thereafter will assume the default encoding to be utf8 (default is cp1252 in Windows)
# This is required for the reverse_geocoder library to work)

import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

# to get the district, country from latitude and longitude data
import reverse_geocoder as rg


# The dataset is of the form .csv , which contains details of the migration data of the bird.
# This includes the tag number of the bird , latitude , longitude , species name and time of the data point4
# The time recorded has been normalised over a year such that the duration is scaled to 1 corresponding to a year. 
# Hence , values closer to zero is the beginning of the year and values closer to 1 
# indicates to the end of the year.

df = pd.read_csv('birds_short.csv')  # reading the csv dataset into a dataframe using Pandas
df2 = pd.read_csv('birds_short.csv')  

sns.set_style(style="white")  # the background style of the plots
plt.rcParams['figure.figsize'] = 14, 12  # Sets the plot to be of 14X12 dimensions 

# the following code disables displaying python and pandas warnings in the terminal
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

##############################################################################################################


def draw_map(species):
    """
    
       Draws a map of the world with points for each bird measurment colored by species.
       
    """
    
    global df2
    birds = df
    # initialize a map
    birdmap = Basemap(lat_0=0, lon_0=0)

    # draw the boundaries and coastlines
    birdmap.drawmapboundary()
    birdmap.drawcoastlines()
    
    # get the list of unique species names
    species = birds.species.unique()
    
    # generate a color for each species
    colors = cm.gist_rainbow(np.linspace(0, 1, len(species)))
    
    
    # for each species, we will plot its locations on the map
    for i, s in enumerate(species):
        
        # extract a df of the species we want using the .loc operator from pandas
        # the arguments to loc are conditions for matches. in this case, we will extract any rows where the 
        # value of 'species' matches the current species 's' we are plotting.
        species_df = birds.loc[birds['species'] == s]
        
        # convert the longitude and latitude from the DataFrame to map positions using the map object
        lo, la = birdmap(species_df['long'], species_df['lat'])
        
        # we use the scatter function of our map object to plot each point. we assign a label, a point size, 's'
        # and a color from our list of colors
        birdmap.scatter(lo, la, label=s, s=2, color=colors[i]) 
       
    # we set a legend for our map. the frameon option draws a border around the legend.
    plt.legend(markerscale=2, frameon=True, loc="lower left", fontsize=12)
    
    print('\n1. Map of migration paths for different bird species: ')
    
    plt.show()
    
    return birdmap

draw_map(df2)

###################################################################################################################


def plot3d_simple(df):
    """
    This plots a simple graph that depicts the path of the bird with the 
    specified tag number in the species we are interested. This is to mainly 
    to understand the visuals of the plots that come up later in the 
    project
    
    """
    # we get a figure object
    fig = plt.figure()
    # we add an Axes3D object to the plot
    ax = fig.add_subplot(111, projection='3d')
    
    # now we can do a scatter plot on the new axes. the c argument colors each point so let's color by time of year
    ax.scatter(df['timeofyear'], df['long'], df['lat'], c=df['timeofyear'])
    
    # Now we set the labels for the plot     
    ax.set_xlabel('Time of year')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Latitude')
    
    print('\n2. Plot of the path of a single bird, in terms of latitude, longitude and time of the year at which it was at that point.')
    
    plt.show()

# To use the function , we must first make a DataFrame , specifically of a bird in the species
# The bird can be specified by using its tag number.

tag = '208'    
d = df2.loc[df2['tag'] == tag ]
plot3d_simple(d)


########################################################################################################################


def cluster(df, key, val):
    """
    
        Takes a dataframe , along with the the key(the parameter on which it must calculate,  like species or tag )
        and the value of the key(Species name or tag number )
        It finds the clusters on longitude and latitude.
        Returns a DataFrame with new column denoting the cluster ID of each point,
        along with the corresponding cluster centers.
        
    """
    
    # For its input , scikit takes a numpy matrix with numerical values. 
    # So we first need to convert our input dataframe accordingly
    # Pandas has a nice function .as_matrix that lets us convert DataFrame columns to numpy matrices
    # But this issues a warning that .as_matrix would be removed in the future , but as of now ,
    # Its still the most convinient to use.
    
    # get DF with the desired bird(s)
    
    s = df.loc[df[key] == val]
    
    df_np = df.as_matrix(columns=['timeofyear', 'long', 'lat'])
    
    # now we can give this data to sickitlearn's KMeans() function.
    # this gives us an object that contains the clustering information such as where the centers are and which points
    # belong to which centers, i.e. labels
    
    kmeans = KMeans(n_clusters=2).fit(df_np)
    
    #store the labels in new DF column 'cluster' , applied on the received DataFrame
    s['cluster'] = kmeans.labels_
    
    return s

df2 = cluster(d, 'tag', tag)


##########################################################################################################################
print('\n3. Plot of co-ordinates of one bird with respect to its proximity to each cluster')

def plot3d_cluster(df, labels=np.array([])):
    """
    
        Now the function will accept two optional arguments: centers, and labels which we will use to visualize
        clustering.
        
        
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(labels) > 0:
        #The DataFrame has supplied the parameter(labels) along which the points are classified.
        ax.scatter(df['timeofyear'], df['long'], df['lat'], c=labels)
    else:
        #Draw a regular plot , with the color shade along the passing time.
        ax.scatter(df['timeofyear'], df['long'], df['lat'], c=df['timeofyear'])
    
        
    ax.set_xlabel('Time of year')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Latitude')

    
    plt.show()
    
    
plot3d_cluster(df2, labels=df2['cluster'])


############################################################################################################################

def migration(df, key, val):
    #d is the dataframe containing the bird we want to work with.
    #Its specified withv the tag number .
    d = df.loc[df[key] == val]
    
    
    #converting the data to a numpy matrix as it requires numerical values.
    X = d.as_matrix(columns=['timeofyear', 'long', 'lat'])
    #do k-means clustering as before
    kmeans = KMeans(n_clusters=2).fit(X)
    #store labels and centers from clustering
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    #print(centers)
    
    #as before, make new column with cluster ID for each point
    d['kmeans'] = labels
    

    #store parameters for gaussian models for each cluster in a dictionary
    gaussians = {}
    
    #get gaussian model parameters
    for i in range(len(centers)):
        #get datapoints belonging to a cluster
        dd = d.loc[d['kmeans'] == i][['timeofyear', 'long', 'lat']]
        #the mean vector is the same as the cluster center vector
        mu = centers[i]
        #compute the covariance matrix using the pandas function cov() that works on a DataFrame
        cov = dd.cov()
        
        #add the model parameters to the dictionary
        gaussians[i] = (mu, cov)

    #assign each point to migrating or not migrating, store 1 if migrating, 0 else
    mig = []
    
    #go through each datapoint with the itertuples() function which gives us an iterator of the rows in the DF
    #the iterator yields a namedtuple object so we can access columns by name
    for i, bird in enumerate(d.itertuples()):
        #get the keys of the gaussian dictionary
        models = gaussians.keys()
        #list to store the distance of each point to each cluster [dist_to_cluster_0, dist_to_cluster_1, ..]
        model_distances = []
        for m in models:
            #get the model parameters for the current cluster
            model = gaussians[m]
            #position vector, will be input to mahalanobis distance
            pos_vec = np.array([bird.timeofyear, bird.long, bird.lat])
            #mean vector
            mu_vec = model[0]
            #inverse of covariance matrix which we convert to a numpy matrix
            S_inv = np.linalg.inv(model[1].as_matrix(columns=['timeofyear', 'long', 'lat']))
            #scipy function for computing mahalanobis distance
            mala = mahalanobis(pos_vec, mu_vec, S_inv)
            #store the distance
            model_distances.append(mala)

        #check if distance above threshold of 1.5 which corresponds to a chi2 probability density of 0.68 
        #this means that points with scores greater than 2 to both clusters 
        #will have a probability of 0.52 of belonging to 
        #the cluster. we use a weak threshold because the data is fairly noisy.
        if model_distances[0] >= 2 and model_distances[1] >= 2:
            mig.append(1)
        else:
            mig.append(0)
    
    #store the migration status in a new column of the DataFrame
    d.loc[:,'migration'] = mig
    
    #make 3D plot using the migration status labels we just obtained
    plot3d_cluster(d, labels=d['migration'])
    
    return d

print('\n4. Plot of co-ordinates of one bird with respect to whether it is migrating or near one of the two bases')
mig = migration(df2, 'tag', tag)


##################################################################################################################


def find_centre(df, key, val):
    """
    
    
        Takes a dataframe and computes clusters on longitude and latitude.
        Returns a numpy array that has the coordinates of the centres of both 
        migration and home base
    
        
        
    """
    
    #The functionality of this is exactly similar to the function cluster() defined previously.
    
    s = df.loc[df[key] == val]
    
    df_np = s.as_matrix(columns=['timeofyear', 'long', 'lat'])
    
    #now we can give this data to sickitlearn's KMeans() function.
    #this gives us an object that contains the clustering information such as where the centers are and which points
    #belong to which centers, i.e. labels
    
    kmeans = KMeans(n_clusters=2).fit(df_np)
    
    #store the labels in new DF column 'cluster'
    s['cluster'] = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    #centers is an array that has the coordinates of the base and migrated phases.
    
    return centers


def locate_base(spec):
    
    """
    This function takes in the name of a species . It then utilises the find_centre() function to calculate the centers
    for every bird in that species. Using this data , we utilise machine learning algorithm to find the generalised
    centre for that particular species
    
    It then uses the draw_base() function to plot the base on a world map.

    """
    location = pd.DataFrame()
    species_list = df.loc[df['species'] == spec]
    
    for bird in species_list['tag'].unique():
        number = bird
        #Finds the centre for the bird
        centers = find_centre(df, 'tag' , number)
        #Add it to the dataframe location.
        location = location.append({'tag' : number , 'timeofyear' :  centers[0][0] , 'long' : centers[0][1] , 'lat' : centers[0][2] } , ignore_index = True)    
        location = location.append({'tag' : number , 'timeofyear' :  centers[1][0] , 'long' : centers[1][1] , 'lat' : centers[1][2] } , ignore_index = True)  
     
        
        
    #Here , we use KMeans to find the general center for the bird species. 
    final =  KMeans(n_clusters=2).fit(location)
    base_centers = final.cluster_centers_
    base = pd.DataFrame()
    
    #Base is a dataframe that has the details of the species's home base.
    base = base.append({'timeofyear' :  base_centers[0][3] , 'long' : base_centers[0][1] , 'lat' : base_centers[0][0] } , ignore_index = True)
    base = base.append({'timeofyear' :  base_centers[1][3] , 'long' : base_centers[1][1] , 'lat' : base_centers[1][0] } , ignore_index = True)
    
   
    draw_base(base)
    

def draw_base(center_data):
    """
        Draws a map of the world with point representing the Home base center.
        Its functionality is again exactly similar to draw_map() defined previously.
        
    """
    
    # initialize a map
    birdmap = Basemap(lat_0=0, lon_0=0)

    # draw the boundaries and coastlines
    birdmap.drawmapboundary()
    birdmap.drawcoastlines()
    
    # convert the longitude and latitude from the DataFrame to map positions using the map object
    lo, la = birdmap(center_data['long'], center_data['lat'])
    
    
    # we use the scatter function of our map object to plot each point. we assign a label, a point size, 's'
    # and a color from our list of colors
    birdmap.scatter(lo, la, facecolor = 'red') 
    
    print('\n6. Plot of the two bases of the Aquila Clanga species')
    plt.show()
    
    # gets the district and country of the centers from the latitude and longitude data
    coordinates = (float(la[0]), float(lo[0])), (float(la[1]), float(lo[1]))
    
    print('\nThe Home Base of the Aquila Clanga species is - ' + 
          str(rg.search(coordinates, mode=1)[0]['name']) + ', ' + 
          str(rg.search(coordinates, mode=1)[0]['admin1']) + 
          '\nThe Migration Base of the Aquila Clanga species is - ' + 
          str(rg.search(coordinates, mode=1)[1]['name']) + ', ' + 
          str(rg.search(coordinates, mode=1)[1]['admin1']))
    
    return birdmap


def show_path(species):
    
    """
    
    This function plots the path of the species under consideration
    
    """

    global df2
    birds = df
    birdmap = Basemap(lat_0=0, lon_0=0)

    #draw the boundaries and coastlines
    birdmap.drawmapboundary()
    birdmap.drawcoastlines()
    
    species_df = birds.loc[birds['species'] == species]
        #convert the longitude and latitude from the DataFrame to map positions using the map object
    lo, la = birdmap(species_df['long'], species_df['lat'])
        #we use the scatter function of our map object to plot each point. we assign a label, a point size, 's'
        #and a color from our list of colors
    birdmap.scatter(lo, la, color="red") 
    
    print('\n5. Plot of the migration path of one species (Aquila Clanga)\n')
    plt.show()
    return birdmap

species = 'Aquila clanga' 
show_path(species)
locate_base(species)


#############################################################################################################