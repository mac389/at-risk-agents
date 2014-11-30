import matplotlib

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist

n = {'nodes':1000,'edges':3}
network_drawing_props = {'cmap':plt.get_cmap('binary'),'width':0.5,'node_size':20,'edge_width':0.5,'with_labels':False}
G = nx.MultiDiGraph()
tmp = nx.barabasi_albert_graph(n['nodes'],n['edges'])
G.add_edges_from(tmp.edges())
G.add_nodes_from(tmp)

#--create mirror image
G.add_edges_from(nx.MultiDiGraph(nx.barabasi_albert_graph(n['nodes'],n['edges'])).reverse().edges())

max_degree = max(nx.degree(G).values())
min_sigmoid = 0.5
max_sigmoid = np.exp(1)/(1.+np.exp(1)) 
degrees = nx.degree(G)
sigmoid = lambda value: (1./(1+np.exp(-value))-min_sigmoid)/(max_sigmoid-min_sigmoid)

for node in degrees:
	tmp = degrees[node]
	degrees[node] = sigmoid(tmp/float(max_degree))

influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}

alpha=degrees
timesteps = 50
#--initial conditions
INITIAL = 0
END=-1
attitudes = np.zeros((n['nodes'],2*timesteps))

THRESHOLDS = np.random.random_sample(size=n['nodes'],)
#--Random attitudes
RANDOM = np.random.random_sample(size=(n['nodes'],))
DRUG_PUSHING = np.random.gamma(2,2,size=(n['nodes'],))
DRUG_PUSHING /= DRUG_PUSHING.max()

attitudes = np.tile(1-DRUG_PUSHING,(2*timesteps,1)).T
normalize = lambda arr: arr/arr.sum()

influence_kernel = {node:normalize(np.array([influence[predecessor] for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

epsilon = 0.1
for t in range(1,timesteps+1):
	for agent in G.nodes():
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*social_influence
		attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)


'''
	#change the susceptibility of the most tolerant agent
	most_tolerant_agent = np.argmax(attitudes[:,t])
	alpha[most_tolerant_agent] = 0
	attitudes[most_tolerant_agent,t] = 1
'''
for t in xrange(timesteps,2*timesteps):
	for agent in G.nodes():
		
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*social_influence
		attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(attitudes[:,INITIAL],color='r',alpha=0.5,bins=20,label='Initial')
ax.hist(attitudes[:,END],color='k',alpha=0.5,bins=20,label='Final')
plt.legend(frameon=False)
plt.show()
