import networkx as nx
import matplotlib.pyplot as plt
#create an instance of an undirected graph using that Graph function
G = nx.Graph()

#add node
G.add_node(1)

G.add_nodes_from([2, 3])

G.add_nodes_from(["u", "v"])

#If we would like to know what are the nodes in our graph,
#we can use the nodes method
G.nodes()
#add edges between node1 and node2, between nodeV and nodeU
G.add_edge(1,2)

G.add_edge("u","v")

#We can add an edge even if the underlying nodes don't already
#exist as part of the graph.
#In that case, Python adds those nodes in automatically.
G.add_edges_from([(1,3), (1,4), (1,5), (1,6)])

G.add_edge("u","w")

G.remove_node(2);
G.remove_nodes_from([4,5])    

G.remove_edge(1,3)        
#remove multiple edges
G.remove_edges_from([(1,2), ("u", "v")])
#find the number of edges
G.number_of_edges()
#find the number of nodes
G.number_of_nodes()

#In this network, the nodes represent members of a karate club and the edges
#correspond to friendships between the members
G = nx.karate_club_graph()

import matplotlib.pyplot as plt

#We first need to import matplotlib pyplot
#as plt. We can now use the nx draw function to visualize our network.
#In this case, I'm going to use a couple of additional keyword arguments.
#First, I would like to have the labels visible inside the nodes.
#To do this, I use the with labels keyword
#and I set that to be equal to true.
#I can also set the node colors and edge colors to be whatever I would like.
#In this case, I'm going to set the node color to be equal to light blue.
#And I'm going to set the edge color to be equal to gray.
#Let's try saving this visualization into a PDF file

nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.savefig("karate_graph.pdf")

G.degree()

#The simplest possible random graph model is the so-called Erdos-Renyi,
#also known as the ER graph model.
#This family of random graphs has two parameters, capital N and lowercase p.
#Here the capital N is the number of nodes in the graph,
#and p is the probability for any pair of nodes to be connected by an edge.
#Here's one way to think about it-- imagine
#starting with N nodes and no edges.
#You can then go through every possible pair of nodes and with probability p
#insert an edge between them.
#In other words, you're considering each pair of nodes
#once, independently of any other pair.
#You flip a coin to see if they're connected,
#and then you move on to the next pair.
#If the value of p is very small, typical graphs generated from the model
#tend to be sparse, meaning having few edges.
#In contrast, if the value of p is large, typical graphs
#tend to be densely connected
#Although the NetworkX library includes an Erdos-Renyi graph generator,
#we'll be writing our own ER function to better understand the model.

N = 20
p = 0.2



from scipy.stats import bernoulli
#In this case, the outcomes are coded as 0s and 1s.
#That means that p is the probability that we get an outcome 1 as opposed
#to outcome 0.The only input argument is p, which is the probability of success
def er_graph(N, p):
    """Generate an ER graph"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G       



nx.draw(er_graph(50, 0.08), node_size=40, node_color="gray")
plt.savefig("er1.pdf")

nx.number_connected_components(er_graph(10, 0)) #number of components

G.number_of_nodes()

#plot the degree distribution for this graph
#I turn this into a list because G.degree.values gives me
#a view object to the values.
#I actually want to create a copy of that and turn it into a list.
def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()), histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$") # the probability of observing a node with that degree
    plt.title("Degree distribution")



G = er_graph(50, 0.08)    
plot_degree_distribution(G)
plt.savefig("hist1.pdf")   
 
    
G1 = er_graph(500, 0.08)    
plot_degree_distribution(G1)
G2 = er_graph(500, 0.08)    
plot_degree_distribution(G2)
G3 = er_graph(500, 0.08)    
plot_degree_distribution(G3)
plt.savefig("hist3.pdf")

# Descriptive Statistics of Empirical Social Networks

import numpy as np
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

#Our next step will be to convert the adjacency matrices to graph objects.
#We will accomplish that by using the to NetworkX graph method.
#So G1, the graph that corresponds to a A1,
#is equal to nx.to networkx graph A1.
#And G2 will be the graph object that is constructed
#from the adjacency matrix called A2

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

#To get a basic sense of the network size and number of connections,
#let's count the number of nodes and the number of edges in the networks.
#In addition, each node has a total number of edges, its degree
#Let's also calculate the mean degree for all nodes in the network.

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    print("Average degree: %.2f" % np.mean(list(G.degree().values())))
    
    
    
basic_net_stats(G1)                     
basic_net_stats(G2)

#find out how large the largest connected component is in our two graphs

#We can extract all components for graph using the following function
#in the NetworkX module
gen = nx.connected_component_subgraphs(G1)

g = gen.__next__() #To get to an actual component, we can use the next method


g.number_of_nodes()

#The size of a component is defined as the number of nodes it contains,
#which as we saw above, we can obtain by applying the len function to a given
#component

G1_LCC = max(nx.connected_component_subgraphs(G1), key=len)
G2_LCC = max(nx.connected_component_subgraphs(G2), key=len)

len(G1_LCC)

#Let's compute the proportion of nodes that
#lie in the largest connected components for these two graphs
G1_LCC.number_of_nodes() / G1.number_of_nodes()

#In practice, it is very common for networks
#to contain one component that encompasses
#a large majority of its nodes, 95, 99, or even 99.9% of all of the nodes
plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray", node_size=20)
plt.savefig("village1.pdf")

plt.figure()
nx.draw(G2_LCC, node_color="green", edge_color="gray", node_size=20)
plt.savefig("village2.pdf")


























