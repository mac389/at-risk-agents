import random, sys,os, json,logging, datetime, itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
plt.switch_backend('Agg')

from numpy.linalg import norm
from actor import actor
from optparse import OptionParser
from scipy.stats import percentileofscore, scoreatpercentile,kruskal
from progressbar import Bar, Percentage, ProgressBar
from termcolor import cprint
from awesome_print import ap
'''
	Remaining experiments: 

	 Validate that what I define as at-risk really is at-risk
'''

'''
	Why is the distribution of intent to drink so narrow?
'''

'''
	For intervention care about the following parameters

	1. Random or targeted
	2. Fraction of network that will receive the message
	3. Beta (sensitivity to this message)
 
    FUTURE THINGS TO MODEL:
     -- Alert fatigue
'''

parser = OptionParser(usage="usage: %prog [options] filename",
                      version="%prog 1.0")
parser.add_option("-t", "--target",
                  action="store",
                  dest="target",
                  default=None,
                  help="Name of destination folder")

parser.add_option("-f", "--fraction",
                  action="store",
                  dest="fraction",
                  default=None,
                  help="Fraction of network to message")

parser.add_option("-r", "--random",
                  action="store_true",
                  dest="random",
                  default=False,
                  help="If passed, 'targets' random nodes")

parser.add_option("-s", "--sensitivity", #Will implement in later version
                  action="store",
                  dest="sensitivity",
                  default=None,
                  help="Sensivity analysis of indicated variable") #This could get messy

(options, args) = parser.parse_args()

#------ Assign more descriptive names to some variables
READ = 'rb'
TAB = '\t'
INITIAL = 0
END=-1
PWD = os.getcwd()

directory = json.load(open('directory.json',READ))
basepath = options.target

logfilename = os.path.join(PWD,basepath,'log-%s.txt'%(datetime.datetime.now().time().isoformat()))
logging.basicConfig(filename=logfilename,level=logging.DEBUG)

params = directory['params']
timesteps = directory['timesteps']
#------ Helper functions
normalize = lambda data: data/np.linalg.norm(data)
hierarchy = {}

######################################################################################
#----------Helper functions
def save(basepath='./'):
	#This function could be more elegant
	#Assumes everything is saved to a directory with a descriptive name. 
	#The directory is created in an overarching shell script and passed to this save function in the variable BASEPATH

	filename = os.path.join(basepath,'attitudes.txt')
	np.savetxt(filename,attitudes,fmt='%.04f',delimiter = TAB)
	logging.info('Saved attitudes as %s'%filename)
	hierarchy['attitudes'] = filename

	filename = os.path.join(basepath,'drinking-behavior.txt')
	np.savetxt(filename,drinking_behavior,fmt='%.04f',delimiter=TAB)
	logging.info('Saved drinking habits as %s'%filename)
	hierarchy['past month drinking'] = filename
	
	filename = os.path.join(basepath,'simulation-record.npy')
	np.save(filename,complete_record)
	hierarchy['complete record'] = filename
	logging.info('Saved complete record as %s'%filename)

	filename = os.path.join(basepath,'alpha.txt')
	np.savetxt(filename,alpha,fmt='%.04f',delimiter=TAB)
	logging.info('Saved alpha as %s'%filename)

	filename = os.path.join(basepath,'at-risk.txt')
	np.savetxt(filename,list(target_idx),fmt='%d',delimiter=TAB)
	hierarchy['at-risk'] = filename
	logging.info('Saved indices of at risk individuals as %s'%filename)

	hierarchyname = os.path.join(PWD,basepath,'directory.json')
	json.dump(hierarchy,open(hierarchyname,'wb'))
	logging.info('Saved a record of all filenames to %s'%hierarchyname)

def identify_at_risk(graph):
	'''
		We hypothesize that an actor is at risk for beginning to use drugs if:
			1. He weighs social influence more heavily than other actors. (His alpha [social susceptibility])
			   is in the upper quartile.)
			2. He receives more highly influential inputs than other actors. (His influence kernel is in the upper quartile with 
				resepct to the number of influencers he recieves where those influencers are the strongest in the network)
			3. Those inputs he weighs highly are from people who consume drugs frequently and have a positive attitude towards
			    the consumption of drugs 
	'''
	all_kernel_values = np.concatenate(influence_kernel.values())

	upper_quartile_influence_kernel = scoreatpercentile(all_kernel_values, 75)
	
	#Calculate distribution of fraction of upper_quartile_influencers per user
	fraction_of_influencers_per_user = [(influence_kernel[agent]>upper_quartile_influence_kernel).sum()/float(len(influence_kernel[agent]))
			for agent in graph.nodes()]

	#Identify those whoe recieve more influencers than other people
	upper_quartile_receiving_influence = scoreatpercentile(fraction_of_influencers_per_user,75)

	#At risk if recent uptick in drinking
	distribution_of_increases_in_drinking = np.diff(drinking_behavior[np.nonzero(drinking_behavior)], axis=1).ravel()
	threshold_for_concerning_drinking = scoreatpercentile(distribution_of_increases_in_drinking,75)
	print threshold_for_concerning_drinking

	upper_quartile_alpha = scoreatpercentile(alpha,75)
	at_risk = [agent for agent in graph.nodes() 
				if alpha[agent]>=upper_quartile_alpha and fraction_of_influencers(influence_kernel[agent],upper_quartile_influence_kernel)>upper_quartile_receiving_influence]
	return at_risk
	

G = nx.MultiDiGraph()
tmp = nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])
G.add_edges_from(tmp.edges())
G.add_nodes_from(tmp)
G.add_edges_from(nx.MultiDiGraph(nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])).reverse().edges())
max_degree = max(nx.degree(G).values())

#Link an agent with each node
actors = [actor() for _ in xrange(params['n']['nodes'])]
for node_idx,actor in zip(G.nodes(),actors):
	G.node[node_idx]['actor'] = actor

alpha=np.random.random_sample(size=(params['n']['nodes'],))
#alpha = np.zeros((params['n']['nodes'],))
influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}
influence_kernel = {node:normalize(np.array([influence[predecessor] 
						for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

fraction = int(options.fraction)*.01
timesteps = directory['timesteps']
total_duration = 3*timesteps

attitudes = np.zeros((params['n']['nodes'],total_duration))
drinking_behavior = np.zeros_like(attitudes)

THRESHOLDS = np.random.random_sample(size=params['n']['nodes'],)

#More efficient computation if tile
INITIAL_ATTITUDES = 2*np.random.random_sample(size=(params['n']['nodes'],))-1
attitudes = np.tile(INITIAL_ATTITUDES,(total_duration,1)).T
INITIAL_ATTITUDES = 2*np.random.random_sample(size=(params['n']['nodes'],))-1
drinking_behavior = np.tile(INITIAL_ATTITUDES,(total_duration,1)).T
complete_record = np.zeros((total_duration,params['n']['nodes'],len(actors[0].variables))).astype(float)
epsilon = directory['epsilon']

beta = 0
start = 1
stop = timesteps

complete_record[0,:,:] =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]
#Save initial distributions
for variable in random.choice(actors).variables.keys():
	tmp = np.array([actor.variables[variable] for actor in actors]) #THIS ISN'T CORRECT
	filename = os.path.join(basepath,'initial-distribution-%s.txt'%(variable.replace(' ','-')))
	np.savetxt(filename,tmp,fmt='%.04f',delimiter=TAB)
	hierarchy[variable] = filename
	logging.info('Saved %s as %s'%(variable,filename))

cprint('Random' if options.random else 'Targeted','red')
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=total_duration).start()

for t in xrange(start,stop):
	agent = random.choice(xrange(params['n']['nodes']))
	internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
	social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
	effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
	attitudes[agent,t] += epsilon*effect

	drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
	local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
												influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

	G.node[agent]['actor'].update({'past month drinking':effect,'attitude to medical consequences':local_medical_attitudes})
	complete_record[t,:,:] =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]

	other_agents = list(set(xrange(params['n']['nodes']))-set([agent]))

	drinking_behavior[other_agents,t] = drinking_behavior[other_agents,t-1] 
	attitudes[other_agents,t] = attitudes[other_agents,t-1]

	pbar.update(t+1)

#---Intervene
start = timesteps
stop = 2*timesteps
beta = 1 #Is distinguishing between susceptibility to different external influences to fine a distinction for this model?
intervention = -.1*params['n']['nodes']

#Calculate distribution of fraction of upper_quartile_influencers per user

#Identify those whoe recieve more influencers than other people
threshold_for_susceptibility = scoreatpercentile(alpha,75)
threshold_for_concerning_drinking = scoreatpercentile(drinking_behavior[:stop].ravel(),75)

at_risk = [agent for agent in G.nodes() if alpha[agent] > threshold_for_susceptibility
				and drinking_behavior[agent,(stop-10):stop].mean()>threshold_for_concerning_drinking]

if options.random:
	target_idx = set(random.sample(xrange(params['n']['nodes']),int(int(options.fraction)*.01*params['n']['nodes']))) 
else:
	target_idx = at_risk


for t in xrange(start,stop):
	agent = random.choice(at_risk)
	internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
	social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
	
	effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
	attitudes[agent,t] += epsilon*effect

	#update agent's drinking behavior
	drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
	local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
												influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

	local_medical_attitudes = (1-beta)*local_medical_attitudes + beta*intervention

	G.node[agent]['actor'].update({'past month drinking':effect,
							       'attitude to medical consequences':local_medical_attitudes})

	
	other_agents = list(set(xrange(params['n']['nodes']))-set([agent]))

	drinking_behavior[other_agents,t] = drinking_behavior[other_agents,t-1] 
	attitudes[other_agents,t] = attitudes[other_agents,t-1]

	pbar.update(t+1)

	complete_record[t,:,:] = np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]

start = 2*timesteps
stop = 3*timesteps

for t in xrange(start,stop):
	agent = random.choice(xrange(params['n']['nodes']))
	internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
	social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
	effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
	attitudes[agent,t] += epsilon*effect

	drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
	local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
												influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

	G.node[agent]['actor'].update({'past month drinking':effect,'attitude to medical consequences':local_medical_attitudes})
	complete_record[t,:,:] =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]

	other_agents = list(set(xrange(params['n']['nodes']))-set([agent]))

	drinking_behavior[other_agents,t] = drinking_behavior[other_agents,t-1] 
	attitudes[other_agents,t] = attitudes[other_agents,t-1]

	pbar.update(t+1)
pbar.finish()

save(basepath=os.path.join(PWD,basepath))