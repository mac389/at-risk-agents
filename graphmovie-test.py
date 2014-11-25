import igraph

import numpy as np 
import matplotlib.pyplot as plt 

from GraphMovie import GraphMovie

n = {'nodes':50,'edges':3}

min_sigmoid = 0.5
max_sigmoid = np.exp(1)/(1.+np.exp(1)) 
sigmoid = lambda value: (1./(1+np.exp(-value))-min_sigmoid)/(max_sigmoid-min_sigmoid)
colorify = lambda color: (255*color,0,0)

'''
Create the GraphMovie object.
'''

m = GraphMovie()

g = igraph.Graph.Barabasi(n['nodes'],n['edges'])
m.addGraph(g)

degrees = np.array(g.degree()).astype(float)
degrees /= degrees.max()
map(sigmoid,degrees)

for id,node in enumerate(g.vs):
     g.vs[id]['color'] = tuple([degrees[id]]*3)
     g.vs[id]['label'] = str(id)

m.addGraph(g)

alpha = degrees

timesteps = 10
#--initial conditions
INITIAL = 0
attitudes = np.zeros((n['nodes'],2*timesteps))
attitudes[:,INITIAL] = np.random.random_sample(size=(n['nodes'],))

for t in xrange(1,timesteps):
    for agent,info in enumerate(g.vs):
        attitudes[agent,t] = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*attitudes[g.neighbors(agent),t-1].mean()
        g.vs[agent]['color'] = colorify(attitudes[agent,t].tolist())
    m.addGraph(g)

#change the susceptibility of the most tolerant agent
most_tolerant_agent = np.argmax(attitudes[:,-1])
alpha[most_tolerant_agent] = 0
attitudes[most_tolerant_agent,-1] = 1

for t in xrange(timesteps,2*timesteps):
    for agent,info in enumerate(g.vs):
        attitudes[agent,t] = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*attitudes[g.neighbors(agent),t-1].mean()
        g.vs[agent]['color'] = colorify(attitudes[agent,t].tolist())
    m.addGraph(g)



'''
Now process the layouts, render the frames, and generate the movie.
'''
m.doMovieLayout()
m.renderMovie()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(attitudes[:,INITIAL],color='r',alpha=0.5,range=(0,1),bins=20,label='Initial')
plt.hold(True)
ax.hist(attitudes[:,timesteps],color='b',alpha=0.5,range=(0,1),bins=20,label='Before polarizer')
ax.hist(attitudes[:,-1],color='k',alpha=0.5,range=(0,1),bins=20,label='After polarizer')
ax.set_ylabel('Frequency')
ax.set_xlabel('Attitude')
plt.legend(frameon=False)
plt.show()