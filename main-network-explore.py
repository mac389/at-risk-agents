import random, sys,os, json,logging, datetime
import numpy as np
import networkx as nx
import visualization as visualization
import matplotlib.pyplot as plt 
from actor import actor
from optparse import OptionParser

'''
	This file is not elegant.
'''


parser = OptionParser(usage="usage: %prog [options] filename",
                      version="%prog 1.0")
parser.add_option("-t", "--target",
                  action="store",
                  dest="target",
                  default=None,
                  help="Name of destination folder")

parser.add_option("-c", "--conditions",
                  action="store",
                  dest="conditions",
                  default=None,
                  help="Name of file with conditions")

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
conditions = [] 
#Look for sensitivity:
if any([len(row.split())>1 for row in open(options.conditions,READ).read().splitlines()]):
	for row in open(options.conditions,READ).read().splitlines():
		item = row.split()
		if len(row.split(' '))>1:
			conditions.append(tuple((item[0],float(item[1]))))
		else:
			conditions.append(tuple((item[0],False)))
total_duration = timesteps*len(conditions)
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

intervention = -1

#Link an agent with each node
actors = [actor() for _ in xrange(params['n']['nodes'])]
for node_idx,actor in zip(G.nodes(),actors):
	G.node[node_idx]['actor'] = actor

alpha=np.random.random_sample(size=(params['n']['nodes'],))
timesteps = directory['timesteps']
toolbar_width = total_duration/directory['toolbar-spacing']

attitudes = np.zeros((params['n']['nodes'],total_duration))
drinking_behavior = np.zeros((params['n']['nodes'],total_duration))


THRESHOLDS = np.random.random_sample(size=params['n']['nodes'],)
RANDOM = np.random.random_sample(size=(params['n']['nodes'],))

attitudes[:,INITIAL] = RANDOM
drinking_behavior[:,INITIAL] = 2*np.random.random_sample(size=(params['n']['nodes'],))-1

#More efficient way to do this?
influence = {node:len(G.predecessors(node))/float(len(G.successors(node))) for node in G.nodes_iter()}
influence_kernel = {node:normalize(np.array([influence[predecessor] for predecessor in G.predecessors(node)]).astype(float))
 						for node in G.nodes_iter()}

mondo_data = np.zeros((total_duration,params['n']['nodes'],len(actors[0].variables))).astype(float)
epsilon = directory['epsilon']

sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
######################################################################################
#--Description of simuations

#BASELINE (to prove no programming errors)
'''
#change the susceptibility of the most tolerant agent
most_tolerant_agent = np.argmax(attitudes[:,t])
alpha[most_tolerant_agent] = 0
attitudes[most_tolerant_agent,t] = 1
'''
PROGRAMMED_CONDITIONS = ['decouple','baseline','educate','educate-sensitivity']
educate = False
######################################################################################
#Allow print statments to go to log
def intervene(condition):
	condition,parameter = condition
	if condition.lower() in PROGRAMMED_CONDITIONS:
		if condition == 'baseline' or condition == 'beta':
			pass
		if condition == 'decouple':
			logging.info('Decoupling by setting alpha of all actors to 0')
			alpha[:] = 0 #Would be nice to propgate the conditions to the graphics
		if condition == 'educate': #Variable of WHOM to educate
			beta = 1
		if condition == 'targeted':
			targeted = True	
	else:
		print 'Condition not recognized, reverting to baseline'
		'''
		print p 
		effect_size = np.std(drinking_behavior[:,END]) - np.std(drinking_behavior[:,INITIAL])
		print effect_size,'effect_size'
		'''
def simulate(condition=None,verbose=False,beta=0,start=0,stop=10,snapshots=False):
	if condition:
		intervene(condition)	
	#Coordinate timing
	logging.info('Condition %s ran from timesteps %d to %d'%(condition,timesteps,2*timesteps))
	if verbose:
		print 'Condition %s ran from timesteps %d to %d'%(condition,timesteps,2*timesteps)

	for t in xrange(start,stop):
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

			effect = tuple((effect,total_medical_attitude))
			G.node[agent]['actor'].update(effect)

		sheet =  np.r_[[G.node[agent]['actor'].snapshot() for agent in G.nodes()]]		

		mondo_data[t,:,:] = sheet

		if t%directory['toolbar-spacing']==0:	
			sys.stdout.write("-")
			sys.stdout.flush()

	if snapshots:
		visualization.snapshots(drinking_behavior[:,start],drinking_behavior[:,stop-1],moniker='beta-%.02f'%beta,basepath=basepath)


#Simulation starts
#Still need to workout timing
for i,condition in enumerate(conditions):
	conditon,parameter = condition
	start = i*directory['timesteps']
	stop = (i+1)*directory['timesteps']-1
	if 'beta' in condition:
		simulate(start=start,stop=stop,condition=condition,beta=float(parameter),snapshots=True) #This is clunky
	else:	
		simulate(start=start,stop=stop,condition=condition,beta=float(parameter)) #This is clunky	
sys.stdout.write("]")
sys.stdout.write("\n")

#--- Create graphical output
save(verbose=False)
visualization.graph_everything(basepath=basepath,moniker=options.target,verbose=False,logfilename=logfilename)
