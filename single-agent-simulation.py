import random, sys,os, json,logging, datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
#plt.switch_backend('Agg')

from numpy.linalg import norm
from actor import actor
from optparse import OptionParser
from scipy.stats import percentileofscore, scoreatpercentile,kruskal
from progressbar import Bar, Percentage, ProgressBar
from termcolor import cprint
from awesome_print import ap 

#(1/100)

n = 100
start = 1 
stop = 20 
actors = [actor() for _ in xrange(10)]
gain = .01
EFFECT_SIZE = -gain * n
normalize = lambda data: data/np.linalg.norm(data)

effect = np.zeros((stop,))
intent = np.zeros_like(effect)
INITIAL_ATTITUDES = 2*np.random.random_sample(size=(n,))-1
attitudes = np.tile(INITIAL_ATTITUDES,(stop,1)).T

G = nx.MultiDiGraph()
tmp = nx.barabasi_albert_graph(n,3)
G.add_edges_from(tmp.edges())
G.add_nodes_from(tmp)
G.add_edges_from(nx.MultiDiGraph(nx.barabasi_albert_graph(n,3)).reverse().edges())
max_degree = max(nx.degree(G).values())

#Link an agent with each node
actors = [actor() for _ in xrange(3)]
for node_idx,actor in zip(G.nodes(),actors):
	G.node[node_idx]['actor'] = actor

alpha=np.random.random_sample(size=(3,))

influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}
influence_kernel = {node:normalize(np.array([influence[predecessor] 
						for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

agent = 0
reps = 20

record = np.zeros((reps,stop))
for rep in range(reps):
	for t in xrange(start,stop):

		EFFECT_SIZE = 0 if t < 10 else -gain*n

		internal_influence = actors[0].calculate_intent_to_drink()
		actors[0].update({'attitude to medical consequences':EFFECT_SIZE})
		ap(actors[0].snapshot(as_dict=True,print_calc=True))
		ap(actors[0].inspect_calculation())
		effect[t] = actors[0].variables['attitude to medical consequences']
		intent[t] = actors[0].variables['intent to drink']


		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		'''
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += (epsilon*(effect if effect > THRESHOLDS[agent] else 0))

		#update agent's drinking behavior
		drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
		local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
													influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

		if agent in target_idx:
			total_external_medical_attitude = beta*intervention + (1-beta)*local_medical_attitudes
		else: 
			total_external_medical_attitude = local_medical_attitudes

		G.node[agent]['actor'].update({'past month drinking':effect,'attitude to medical consequences':local_medical_attitudes})
		'''
	record[rep,:] = intent

fig = plt.figure() 
ax = fig.add_subplot(111)
#ax.plot(effect,'k',linewidth=2)
plt.hold(True)
ax.hist(record.mean(axis=0),weights=np.ones_like(record.mean(axis=0))/len(record.mean(axis=0)))
#ax.errorbar(range(record.shape[1]),record.mean(axis=0),yerr=record.std(axis=0)/record.shape[0],fmt='--o')
plt.tight_layout()
plt.show()
