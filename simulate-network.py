import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
n = {'nodes':20,'edges':3}

G = nx.barabasi_albert_graph(n['nodes'],n['edges'])

max_degree = max(nx.degree(G).values())
min_sigmoid = 0.5
max_sigmoid = np.exp(1)/(1.+np.exp(1)) 
degrees = nx.degree(G)
sigmoid = lambda value: (1./(1+np.exp(-value))-min_sigmoid)/(max_sigmoid-min_sigmoid)

for node in degrees:
	tmp = degrees[node]
	degrees[node] = sigmoid(tmp/float(max_degree))



#Choose alpha from a uniform random distirbution 
alpha = degrees

timesteps = 10
#--initial conditions
INITIAL = 0
attitudes = np.zeros((n['nodes'],timesteps))
attitudes[:,INITIAL] = np.random.random_sample(size=(n['nodes'],))

#TODO: Influence kernel
for node in G.nodes():
	print G.successors(node)


for t in xrange(1,timesteps):
	for agent in G.nodes():
		attitudes[agent,t] = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*attitudes[G.neighbors(agent),t-1].mean()