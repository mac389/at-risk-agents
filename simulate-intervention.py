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
######################################################################################
#----------Helper functions
def save(basepath='./'):
	#This function could be more elegant
	#Assumes everything is saved to a directory with a descriptive name. 
	#The directory is created in an overarching shell script and passed to this save function in the variable BASEPATH
	hierarchy = {}

	for variable in random.choice(actors).variables.keys():
		tmp = np.array([actor.variables[variable] for actor in actors])
		filename = os.path.join(basepath,'initial-distribution-%s.txt'%(variable.replace(' ','-')))
		np.savetxt(filename,tmp,fmt='%.04f',delimiter=TAB)
		hierarchy[variable] = filename
		logging.info('Saved %s as %s'%(variable,filename))

	filename = os.path.join(basepath,'attitudes.txt')
	np.savetxt(filename,attitudes,fmt='%.04f',delimiter = TAB)
	logging.info('Saved attitudes as %s'%filename)
	hierarchy['attitudes'] = filename

	filename = os.path.join(basepath,'drinking-behavior.txt')
	np.savetxt(filename,drinking_behavior,fmt='%.04f',delimiter=TAB)
	logging.info('Saved drinking habits as %s'%filename)
	hierarchy['past month drinking'] = filename
	
	filename = os.path.join(basepath,'simulation-record--%s--.txt'%('-'.join(map(str,complete_record.shape))))
	np.savetxt(filename,complete_record.flatten(),fmt='%.04f',delimiter=TAB)
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

	#Make this more efficient
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
	
def recent_running_average(arr,idx,period):
	data = arr[idx,:]
	data = data[np.nonzero(data)]
	return data[-period:].mean()

def fraction_of_concerning_influencers(graph,agent):
	influencers = graph.predecessors(agent)

def identify_at_risk(graph):
	return [agent for agent in graph.nodes() if is_at_risk(graph,agent)]

def is_at_risk(graph, agent):
	'''
		We hypothesize that an actor is at risk for beginning to use drugs if:
			1. He weighs social influence more heavily than other actors. (His alpha [social susceptibility])
			   is in the upper quartile.)
			2. He receives more highly influential inputs than other actors. (His influence kernel is in the upper quartile with 
				resepct to the number of influencers he recieves where those influencers are the strongest in the network)
			3. Those inputs he weighs highly are from people who consume drugs frequently and have a positive attitude towards
			    the consumption of drugs 
	'''
	return all([alpha[agent] > thresdhold_for_susceptibility,recent_running_average(drinking_behavior,agent,4) > threshold_for_concerning_hx_of_drinking,
		fraction_of_influencers(influence_kernel[agent],threshold_for_concerning_influencers)>threshold_for_number_of_concerning_influencers])

def fraction_of_influencers(arr,cutoff):
	#Assuming input is 1D numpy array
	if not isinstance(arr,np.ndarray):
		arr = np.array(arr)
	return (arr>cutoff).sum()/float(len(arr))

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

influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}
influence_kernel = {node:normalize(np.array([influence[predecessor] 
						for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

fraction = int(options.fraction)*.01
#Previously was identify_at_risk(G); however, updated definition of at-risk to include drinking behavior and so 
#cannot determine at-risk before simulation begins
timesteps = directory['timesteps']
total_duration = 2*timesteps

attitudes = np.zeros((params['n']['nodes'],total_duration))
drinking_behavior = np.zeros_like(attitudes)

THRESHOLDS = np.random.random_sample(size=params['n']['nodes'],)

#More efficient computation if tile
INITIAL_ATTITUDES = 2*np.random.random_sample(size=(params['n']['nodes'],))-1
attitudes = np.tile(INITIAL_ATTITUDES,(total_duration,1)).T
drinking_behavior[:,INITIAL] = 2*np.random.random_sample(size=(params['n']['nodes'],))-1
complete_record = np.zeros((total_duration,params['n']['nodes'],len(actors[0].variables))).astype(float)
epsilon = directory['epsilon']

beta = 0
start = 1
stop = timesteps

cprint('Random' if options.random else 'Targeted','red')
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=total_duration).start()
for t in xrange(start,stop):
	for agent in G.nodes():
		
		internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += (epsilon*(effect if effect > THRESHOLDS[agent] else 0))

		drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
		local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
													influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

		G.node[agent]['actor'].update({'past month drinking':effect,'attitude to medical consequences':local_medical_attitudes})
	complete_record[t,:,:] =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]

	pbar.update(t+1)

#---Intervene
start = timesteps
stop = total_duration
beta = 1 #Is distinguishing between susceptibility to different external influences to fine a distinction for this model?
intervention = .01*params['n']['nodes']

#Calculate distribution of fraction of upper_quartile_influencers per user

#Identify those whoe recieve more influencers than other people
thresdhold_for_susceptibility = scoreatpercentile(alpha,75)
threshold_for_concerning_influencers = scoreatpercentile(np.concatenate(influence_kernel.values()),75)

fraction_of_concerning_influencers_per_user = [(influence_kernel[agent]>fraction_of_concerning_influencers).sum()/float(len(influence_kernel[agent]))
			for agent in G.nodes()]
threshold_for_number_of_concerning_influencers = scoreatpercentile(fraction_of_concerning_influencers_per_user,75)

distribution_of_increases_in_drinking =  np.diff(drinking_behavior,axis=1).ravel()
distribution_of_increases_in_drinking = distribution_of_increases_in_drinking[np.nonzero(distribution_of_increases_in_drinking)]
threshold_for_concerning_hx_of_drinking = scoreatpercentile(distribution_of_increases_in_drinking,75)
if options.random:
	target_idx = set(random.sample(xrange(params['n']['nodes']),int(int(options.fraction)*.01*params['n']['nodes']))) 
else:
	target_idx = set(identify_at_risk(G))

for t in xrange(start,stop):
	for agent in G.nodes():
		
		internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += (epsilon*(effect if effect > THRESHOLDS[agent] else 0))

		#update agent's drinking behavior
		drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
		local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
													influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

		if (options.random and agent in target_idx) or (not options.random and agent in target_idx):
			total_external_medical_attitude = beta*intervention + (1-beta)*local_medical_attitudes
		else: 
			total_external_medical_attitude = local_medical_attitudes

		G.node[agent]['actor'].update({'past month drinking':effect,'attitude to medical consequences':local_medical_attitudes,
			'self-actualizing values':effect})

	complete_record[t,:,:] = np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]

	pbar.update(t+1)
pbar.finish()

#--measure effect
before = attitudes[:,:timesteps]
after = attitudes[:,timesteps:]

before_points = zip(before.std(axis=0),before.mean(axis=0))
after_points = zip(after.std(axis=0),after.mean(axis=0))

effect_size = sum([np.sqrt((after_point[0]-before_point[0])**2 + (after_point[1]-before_point[1])**2)
				for before_point,after_point in zip(before_points,after_points)])

effect_size /= float(total_duration)
#Should I calculate distance in CDF space?
with open('./effect_size_random' if options.random else './effect_size','a+') as out:
	print>>out,'%.04f \t %.04f\n'%(effect_size, fraction), 

save(basepath=os.path.join(PWD,basepath))