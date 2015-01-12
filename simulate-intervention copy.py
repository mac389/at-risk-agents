import random, sys,os, json,logging, datetime
import numpy as np
import networkx as nx
import visualization as visualization
import matplotlib.pyplot as plt 

from numpy.linalg import norm
from actor import actor
from optparse import OptionParser
from scipy.stats import percentileofscore, scoreatpercentile,kruskal

'''
	For intervention care about the following parameters

	1. Random or targeted
	2. Fraction of network that will receive the message
 
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

directory = json.load(open('directory.json',READ))
#--Touch diredctory, could do this in BASH, there is no error checking here
if not os.path.exists(options.target):
    os.makedirs(options.target)
basepath = options.target
logfilename = os.path.join(os.getcwd(),basepath,'log-%s.txt'%(datetime.datetime.now().time().isoformat()))
logging.basicConfig(filename=logfilename,level=logging.DEBUG)

params = directory['params']
timesteps = directory['timesteps']
#------ Helper functions
normalize = lambda data: (data-data.min())/float(data.max()-data.min())


######################################################################################
#----------Helper functions
def save(basepath=basepath,verbose=True):
	#This function could be more elegant
	#Assumes everything is saved to a directory with a descriptive name. 
	#The directory is created in an overarching shell script and passed to this save function in the variable BASEPATH
	hierarchy = {}

	for variable in random.choice(actors).variables.keys():
		tmp = np.array([actor.variables[variable] for actor in actors])
		filename = os.path.join(os.getcwd(),basepath,'initial-distribution-%s.txt'%(variable.replace(' ','-')))
		np.savetxt(filename,tmp,fmt='%.04f',delimiter=TAB)
		if verbose:
			print 'Saved %s'%variable
			hierarchy[variable] = filename
		logging.info('Saved %s as %s'%(variable,filename))
	filename = os.path.join(os.getcwd(),basepath,'attitudes.txt')
	np.savetxt(filename,attitudes,fmt='%.04f',delimiter = TAB)
	if verbose:
		print 'Saved attitudes'
	logging.info('Saved attitudes as %s'%filename)
	hierarchy['attitudes'] = filename
	filename = os.path.join(os.getcwd(),basepath,'drinking-behavior.txt')
	np.savetxt(filename,drinking_behavior,fmt='%.04f',delimiter=TAB)
	if verbose:
		print 'Saved drinking habits'
	logging.info('Saved drinking habits as %s'%filename)
	hierarchy['past month drinking'] = filename
	filename = os.path.join(os.getcwd(),basepath,'simulation-record--%s--.txt'%('-'.join(map(str,mondo_data.shape))))
	np.savetxt(filename,mondo_data.flatten(),fmt='%.04f',delimiter=TAB)
	if verbose:
		print 'Saved complete record'
	hierarchy['complete record'] = filename
	logging.info('Saved complete record as %s'%filename)

	filename = os.path.join(os.getcwd(),basepath,'alpha.txt')
	np.savetxt(filename,alpha,fmt='%.04f',delimiter=TAB)
	if verbose:
		print 'Saved social susceptibility'
	logging.info('Saved alpha as %s'%filename)

	hierarchyname = os.path.join(os.getcwd(),basepath,'directory.json')
	json.dump(hierarchy,open(hierarchyname,'wb'))
	logging.info('Saved a record of all filenames to %s'%hierarchyname)


def identify_at_risk(graph):
	'''
		We hypothesize that an actor is at risk for beginning to use drugs if:
			1. He weighs social influence more heavily than other actors. (His alpha [social susceptibility])
			   is in the upper quartile.)
			2. He receives more highly influential inputs than other actors. (His influence kernel is in the upper quartile with 
				resepct to the number of influencers he recieves where those influencers are the strongest in the network)
	'''
	all_kernel_values = np.concatenate(influence_kernel.values())
	upper_quartile_influence_kernel = scoreatpercentile(all_kernel_values, 75)
	
	#Calculate distribution of fraction of upper_quartile_influencers per user
	fraction_of_influencers_per_user = [(influence_kernel[agent]>upper_quartile_influence_kernel).sum()/float(len(influence_kernel[agent]))
			for agent in graph.nodes()]

	#Identify those whoe recieve more influencers than other people
	upper_quartile_receiving_influence = scoreatpercentile(fraction_of_influencers_per_user,75)

	upper_quartile_alpha = scoreatpercentile(alpha,75)
	at_risk = [agent for agent in graph.nodes() 
		if alpha[agent]>=upper_quartile_alpha and fraction_of_influencers(influence_kernel[agent],upper_quartile_influence_kernel)>upper_quartile_receiving_influence]
	return at_risk
	
def fraction_of_influencers(arr,cutoff):
	#Assuming input is 1D numpy array
	if not isinstance(arr,np.ndarray):
		arr = np.array(arr)
	return (arr>cutoff).sum()/float(len(arr))


def baseline():
	logging.info('Baseline simulation from timesteps 1 to %d'%timesteps)
	for t in range(1,timesteps):
		for agent in G.nodes():
			internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
			social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
			effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
			attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)

			#update agent's drinking behavior
			drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
			local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
														influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

			local_medical_attitudes /= np.sum(influence_kernel[agent])
			total_medical_attitude = beta*(intervention) + (1-beta)*local_medical_attitudes

			effect = tuple((effect,local_medical_attitudes))
			G.node[agent]['actor'].update(effect)

		sheet =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]		
		mondo_data[t,:,:] = sheet	

		if t%directory['toolbar-spacing']==0:	
			sys.stdout.write("-")
			sys.stdout.flush()

G = nx.MultiDiGraph()
tmp = nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])
G.add_edges_from(tmp.edges())
G.add_nodes_from(tmp)
actors = [actor() for _ in xrange(params['n']['nodes'])]
G.add_edges_from(nx.MultiDiGraph(nx.barabasi_albert_graph(params['n']['nodes'],params['n']['edges'])).reverse().edges())
max_degree = max(nx.degree(G).values())

#Link an agent with each node
actors = [actor() for _ in xrange(params['n']['nodes'])]
for node_idx,actor in zip(G.nodes(),actors):
	G.node[node_idx]['actor'] = actor

alpha=np.random.random_sample(size=(params['n']['nodes'],))

#More efficient way to do this?
influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}
influence_kernel = {node:normalize(np.array([influence[predecessor] for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

fraction = int(options.fraction)*.01
print "FRACTION ", fraction
if options.random:
	target_idx = set(random.sample(xrange(params['n']['nodes']),int(int(options.fraction)*.01*params['n']['nodes'])))
else:
	target_idx = set(identify_at_risk(G))

timesteps = directory['timesteps']
total_duration = 2*timesteps
toolbar_width = total_duration/directory['toolbar-spacing']

attitudes = np.zeros((params['n']['nodes'],total_duration))
drinking_behavior = np.zeros((params['n']['nodes'],total_duration))

THRESHOLDS = np.random.random_sample(size=params['n']['nodes'],)
RANDOM = np.random.random_sample(size=(params['n']['nodes'],))

attitudes[:,INITIAL] = RANDOM
drinking_behavior[:,INITIAL] = 2*np.random.random_sample(size=(params['n']['nodes'],))-1

mondo_data = np.zeros((total_duration,params['n']['nodes'],len(actors[0].variables))).astype(float)
epsilon = directory['epsilon']


sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

beta = 0
snapshots = True
for t in xrange(1,timesteps):
	start = 1
	stop = timesteps
	for agent in G.nodes():
		
		internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)

		#update agent's drinking behavior
		drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
		local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
													influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

		local_medical_attitudes /= np.sum(influence_kernel[agent])

		effect = tuple((effect,local_medical_attitudes))
		G.node[agent]['actor'].update(effect)

	sheet =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]		

	mondo_data[t,:,:] = sheet

	if t%directory['toolbar-spacing']==0:	
		sys.stdout.write("-")
		sys.stdout.flush()

if snapshots:
	visualization.snapshots(drinking_behavior[:,start],drinking_behavior[:,stop-1],moniker='beta-%.02f'%beta,basepath=basepath)

beta = 0.5 #Is distinguishing between susceptibility to different external influences to fine a distinction for this model?
intervention = -1
#---Intervene
for t in xrange(timesteps,total_duration):
	start = timesteps
	stop = total_duration
	for agent in G.nodes():
		
		internal_influence = G.node[agent]['actor'].calculate_intent_to_drink()
		social_influence = attitudes[G.predecessors(agent),t-1].dot(influence_kernel[agent]) #kernel already normalized
		effect = (1-alpha[agent])*internal_influence + alpha[agent]*social_influence
		attitudes[agent,t] += epsilon*(effect if effect > THRESHOLDS[agent] else 0)

		#update agent's drinking behavior
		drinking_behavior[agent,t] = G.node[agent]['actor'].variables['past month drinking']
		local_medical_attitudes = np.array([G.node[influencer]['actor'].variables['attitude to medical consequences'] for 
													influencer in G.predecessors(agent)]).dot(influence_kernel[agent])

		local_medical_attitudes /= np.sum(influence_kernel[agent])
		if agent in target_idx:
			total_external_medical_attitude = beta*intervention + (1-beta)*local_medical_attitudes
		else: 
			total_external_medical_attitude = local_medical_attitudes
		effect = tuple((effect,total_external_medical_attitude))
		G.node[agent]['actor'].update(effect)

	sheet =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]		

	mondo_data[t,:,:] = sheet

	if t%directory['toolbar-spacing']==0:	
		sys.stdout.write("-")
		sys.stdout.flush()

if snapshots:
	visualization.snapshots(drinking_behavior[:,start],drinking_behavior[:,stop-1],moniker='beta-%.02f'%beta,basepath=basepath)

sys.stdout.write("]")
sys.stdout.write("\n")

#--measure effect
before = attitudes[:,:timesteps]
after = attitudes[:,timesteps:]

before_points = zip(before.std(axis=0),before.mean(axis=0))
after_points = zip(after.std(axis=0),after.mean(axis=0))

effect_size = sum([np.sqrt((after_point[0]-before_point[0])**2 + (after_point[1]-before_point[1])**2)
				for before_point,after_point in zip(before_points,after_points)])

effect_size /= float(total_duration)
#Should I calculate distance in CDF space?
effect_file = './effect_size_random' if options.random else './effect_size'
with open(effect_file,'a+') as out:
	print>>out,'%.04f \t %.04f\n'%(effect_size, fraction), 

#--- Create graphical output
save(verbose=False)
visualization.graph_everything(basepath=basepath,moniker=options.target,verbose=False,logfilename=logfilename)

#Need a measure to show their behaviors are different
visualization.population_summary(moniker=options.target+'-at-risk',basepath=basepath,criterion=list(target_idx), criterionname='at risk')
visualization.time_series(moniker=options.target, basepath=basepath,criterion = list(target_idx),
	criterionname='at risk')