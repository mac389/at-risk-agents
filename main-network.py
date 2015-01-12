import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import Graphics as artist

from matplotlib import rcParams
from actor import actor

rcParams['text.usetex'] = True


params = {'n':{'nodes':1000,'edges':3},
		'network_drawing_props':{'cmap':plt.get_cmap('binary'),'width':0.5,'node_size':20,'edge_width':0.5,'with_labels':False},
		 'hist_props' : {'color':'k','range':(0,1),'histtype':'stepfilled'}}

G = nx.MultiDiGraph()
tmp = nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])
G.add_edges_from(tmp.edges())
G.add_nodes_from(tmp)

actors = [actor() for _ in xrange(params['n']['nodes'])]

nrows = 2
ncols = 3

#--create mirror image
G.add_edges_from(nx.MultiDiGraph(nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])).reverse().edges())

max_degree = max(nx.degree(G).values())
min_sigmoid = 0.5
max_sigmoid = np.exp(1)/(1.+np.exp(1)) 
degrees = nx.degree(G)
sigmoid = lambda value: (1./(1+np.exp(-value))-min_sigmoid)/(max_sigmoid-min_sigmoid)

for node in degrees:
	tmp = degrees[node]
	degrees[node] = sigmoid(tmp/float(max_degree))

influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}


#Link an agent with each node
actors = [actor() for _ in xrange(params['n']['nodes'])]
for node_idx,actor in zip(G.nodes(),actors):
	G.node[node_idx]['actor'] = actor

alpha=degrees
timesteps = 5
#--initial conditions
INITIAL = 0
END=-1
attitudes = np.zeros((params['n']['nodes'],2*timesteps))

THRESHOLDS = np.random.random_sample(size=params['n']['nodes'],)
#--Random attitudes
RANDOM = np.random.random_sample(size=(params['n']['nodes'],))
DRUG_PUSHING = np.random.gamma(2,2,size=(params['n']['nodes'],))
DRUG_PUSHING /= DRUG_PUSHING.max()

attitudes = np.tile(1-DRUG_PUSHING,(2*timesteps,1)).T
normalize = lambda arr: arr/arr.sum()

influence_kernel = {node:normalize(np.array([influence[predecessor] for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

print influence_kernel
epsilon = 0.1
for t in range(1,timesteps+1):
	for agent in G.nodes():
		internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)

		#update agent's drinking behavior
		G.node[agent]['actor'].update(effect)

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

initial_db = True

if initial_db:
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)
	yvars = random.choice(actors).variables.keys()
	for i,col in enumerate(axs):
		for j,row in enumerate(col):
			axs[i,j].hist([actor.variables[yvars[i*ncols+j]] for actor in actors],**params['hist_props'])
			fig.canvas.mpl_connect('draw_event', artist.on_draw)
			artist.adjust_spines(axs[i,j])
			if 'attitude' not in yvars[i*ncols+j]: 	
				axs[i,j].set_xlabel(artist.format(yvars[i*ncols+j]))
			elif 'psychological' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude to','psychological','consequences']))
				axs[i,j].set_xlabel(label)
			elif 'medical' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude','to medical','consequences']))
				axs[i,j].set_xlabel(label)
	plt.tight_layout()
	plt.savefig('dashboard-f.png',dpi=300)