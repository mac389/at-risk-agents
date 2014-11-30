import networkx as nx
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import Graphics as artist
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Changing attitudes in a social network', artist='ToxTweet',
        comment='TBW')
writer = FFMpegWriter(fps=5, metadata=metadata)

n = {'nodes':20,'edges':3}
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

gs = gridspec.GridSpec(1,2,width_ratios=[3,1])

fig = plt.figure()
ax = plt.subplot(gs[0])
hist_panel = plt.subplot(gs[1])
alpha=degrees
timesteps = 50
#--initial conditions
INITIAL = 0
attitudes = np.zeros((n['nodes'],2*timesteps))
attitudes[:,INITIAL] = np.random.random_sample(size=(n['nodes'],))

with writer.saving(fig, "blah.mp4", 100):
	nx.draw(G,pos=nx.circular_layout(G), node_color=[degrees[node] for node in G.nodes()],ax=ax, **network_drawing_props)
	for t in range(1,timesteps+1):
		for agent in G.nodes():
			
			#Influence kernel can DEFINITELY be precomputed
			influence_kernel = np.array([influence[predecessor] for predecessor in G.predecessors(agent)]).astype(float)
			social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel)
			social_influence /= influence_kernel.sum()

			attitudes[agent,t] = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*social_influence
		nx.draw_networkx(G,pos=nx.circular_layout(G), node_color=[attitudes[node,t] for node in G.nodes()],
			ax=ax,**network_drawing_props)
		hist_panel.hist(attitudes[:,INITIAL],color='r',alpha=0.5,range=(0,1),bins=20,label='Initial')
		hist_panel.hist(attitudes[:,t],color='k',alpha=0.5,range=(0,1),bins=20,label='Current')
		artist.adjust_spines(hist_panel)
		hist_panel.axvline(x=attitudes[:,INITIAL].mean(),color='r',linewidth=2,linestyle='--')
		hist_panel.axvline(x=attitudes[:,t].mean(),color='k',linewidth=2,linestyle='--')
		hist_panel.set_ylabel('Frequency')
		hist_panel.set_ylim(ymin=0,ymax=50)
		hist_panel.set_xlabel('Attitude')
		#plt.legend(frameon=False)
		writer.grab_frame()
		hist_panel.cla()

	#change the susceptibility of the most tolerant agent
	most_tolerant_agent = np.argmax(attitudes[:,t])
	alpha[most_tolerant_agent] = 0
	attitudes[most_tolerant_agent,t] = 1
	for t in xrange(timesteps,2*timesteps):
		for agent in G.nodes():
			influence_kernel = np.array([influence[predecessor] for predecessor in G.predecessors(agent)]).astype(float)
			social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel)
			social_influence /= influence_kernel.sum()

			attitudes[agent,t] = (1-alpha[agent])*attitudes[agent,t-1] + alpha[agent]*social_influence
		nx.draw_networkx(G,pos=nx.circular_layout(G), node_color=[attitudes[node,t] for node in G.nodes()], 
			ax=ax, **network_drawing_props)
		hist_panel.hist(attitudes[:,INITIAL],color='r',alpha=0.5,range=(0,1),bins=20,label='Initial')
		hist_panel.hist(attitudes[:,t],color='k',alpha=0.5,range=(0,1),bins=20,label='Current')
		artist.adjust_spines(hist_panel)
		hist_panel.axvline(x=attitudes[:,INITIAL].mean(),color='r',linewidth=2,linestyle='--')
		hist_panel.axvline(x=attitudes[:,t].mean(),color='k',linewidth=2,linestyle='--')
		hist_panel.set_ylabel('Frequency')
		hist_panel.set_ylim(ymin=0,ymax=50)
		hist_panel.set_xlabel('Attitude')
		#plt.legend(frameon=False)
		writer.grab_frame()
		hist_panel.cla()
