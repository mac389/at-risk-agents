import os, json,re, datetime, itertools

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt 
import visualization as visualization

from awesome_print import ap
from matplotlib import rcParams
from optparse import OptionParser
from scipy.stats import percentileofscore,scoreatpercentile
from texttable import Texttable
from scipy.stats import ks_2samp


plt.switch_backend('Agg')

params = {
   'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True
   }

rcParams.update(params)

parser = OptionParser(usage="usage: %prog [options] filename",
                      version="%prog 1.0")
parser.add_option("-s", "--source",
                  action="store",
                  dest="source",
                  default=False,
                  help="Folder with data to analyze")
(options, args) = parser.parse_args()

READ = 'rb'
TAB = '\t'
INTEGER = '%d'
FLOAT = '%.04f'
hist_props={"range":[-1,1],"histtype":"stepfilled"}
make_filename = lambda filename: os.path.join(os.getcwd(),basepath,filename)
basepath = os.path.join(os.getcwd(),options.source)
logfilename = os.path.join(basepath,'log-%s.txt'%(datetime.datetime.now().time().isoformat()))
USERS = 0
TIME = 1
PAST_MONTH_DRINKING = 3

#---------HELPER FUNCTIONS
def compare_demographics(data,nrows=2,ncols=3):
	#Data is a list of tuples of (label,data,color)
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=True)
	first_label,first_idx,first_color = data[0]
	second_label,second_idx,second_color = data[1]

	MALE = 0.5
	FEMALE = 0.3
	for i,col in enumerate(axs):
		for j,row in enumerate(col):
				characteristic = characteristics[i*ncols+j]
				uq = demographics[characteristic][first_idx]
				lq = demographics[characteristic][second_idx]
						
				_,_,patches1=row.hist(uq,color=first_color,label=artist.format(first_label), histtype='step',
					weights = np.ones_like(uq)/len(uq))
				plt.hold(True)
				_,_,patches2=row.hist(lq,color=second_color,label=artist.format(second_label),histtype='step',
					weights=np.ones_like(lq)/len(lq))
				fig.canvas.mpl_connect('draw_event', artist.on_draw)
				artist.adjust_spines(row)
				if 'attitude' not in yvars[i*ncols+j]: 	
					row.set_xlabel(artist.format(yvars[i*ncols+j]))
					if 'gender' in yvars[i*ncols+j]:
						axs[i,j].set_xticks([FEMALE,MALE])
						axs[i,j].set_xticklabels(map(artist.format,['Female','Male']))
				elif 'psychological' in yvars[i*ncols+j]:
					label = '\n'.join(map(artist.format,['Attitude to','psychological','consequences']))
					row.set_xlabel(label)
				elif 'medical' in yvars[i*ncols+j]:
					label = '\n'.join(map(artist.format,['Attitude','to medical','consequences']))
					row.set_xlabel(label)
					#axs[i,j].set_xlim([-50,50])

	plt.tight_layout()
	fig.legend((patches1[0], patches2[0]), (artist.format(first_label),artist.format(second_label)),
		loc='lower right', frameon=False, ncol=2)
	#filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-no-temporal-threshold.png')
	filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-%s-vs-%s.png'%(first_label,second_label))
	plt.savefig(filename,dpi=300)
	del fig,axs,i,j

def compare_distributions(variable_source_name,idxs,rng=(0,1)):
	#Assume idxs is dictionary structured as {name:[corresponding indices]}
	fig = plt.figure()
	ax = fig.add_subplot(111)
	data =  np.loadtxt(make_filename('%s.txt'%(variable_source_name)),delimiter=TAB)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.hold(True)
	for subpopulation,idx,color in idxs:
		weights = np.ones_like(data[idx])/len(data[idx])
		ax.hist(data[idx],color=color,label=artist.format(subpopulation),histtype='step',range=rng,weights=weights)

	fig.canvas.mpl_connect('draw_event', artist.on_draw)
	artist.adjust_spines(ax)
	ax.set_ylabel(artist.format('Prevalance'))
	ax.set_xlabel(artist.format(variable_source_name))
	plt.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.05))
	plt.tight_layout()
	plt.savefig(make_filename('%s-%s.png'%(variable_source_name,'-'.join([idx[0] for idx in idxs]))),dpi=300)
	del fig,ax


#-----INITIALIZE------------------------------------------
data = {}
directory = json.load(open(os.path.join(basepath,'directory.json'),READ))
for variable in directory:
	data[variable] = np.load(directory[variable]) if variable == 'complete record' else np.loadtxt(directory[variable],delimiter = TAB)


RESPONDER_FILENAME = os.path.join(basepath,'responders')
if not os.path.isfile(RESPONDER_FILENAME):

	responders = [agent for agent in xrange(data['complete record'].shape[1])
					if np.gradient(np.array_split(data['complete record'][:,agent,PAST_MONTH_DRINKING],3)[1]).mean()<0]

	np.savetxt(RESPONDER_FILENAME,responders,delimiter=TAB,fmt=INTEGER) 
	ap('%d Responders: %s'%(len(responders),' '.join(map(str,responders))))
	identified_responders = set(responders) & set(data['at-risk'])
	ap('%d Responders identified as at-risk: %s'%(len(identified_responders),map(str,identified_responders)))

else:
	responders = np.loadtxt(RESPONDER_FILENAME,delimiter=TAB)

overall_population = data['attitudes'].shape[0]
yes_response_yes_atrisk = len(set(responders) & set(data['at-risk']))
no_response_yes_atrisk = len(set(data['at-risk']) - set(responders))
no_response_no_atrisk =  len(set(range(overall_population)) - set(responders)-set(data['at-risk']))
yes_response_no_atrisk =  len(set(responders)-set(data['at-risk'])) 

#print contingency_table
table = Texttable()
table.set_cols_align(["r", "l","l","l"])
table.set_cols_valign(["t", "b","b","b"])

table.add_rows([ ["","", "At-risk", ""],
				 ["","","+","-"],  
        		 ["Responded","+", yes_response_yes_atrisk, yes_response_no_atrisk],
        		 ["","-",no_response_yes_atrisk,no_response_no_atrisk]])
print(table.draw() + "\n")

try:
	print 'Sensitivity: %.02f'%(yes_response_yes_atrisk/float(yes_response_yes_atrisk+no_response_yes_atrisk))
except:
	print "Sensitivity: -1"
try:
	print 'Specificity: %.02f'%(no_response_no_atrisk/float(yes_response_no_atrisk+no_response_no_atrisk))
except:
	print "Sensitivity -1"

#Do the heaviest consumers have different demographics than the lightest consumers?
upper_quartile_cutoff = scoreatpercentile(data['past month drinking'],75)
lower_quartile_cutoff = scoreatpercentile(data['past month drinking'],25)

#ap('Upper drinking cutoff: %.02f, lower cutoff %.02f'%(upper_quartile_cutoff,lower_quartile_cutoff))

#When loading complete record from file, first axis is time, second axis is agent, third axis is variable
light_users_idx = np.where(data['complete record'][:,:,PAST_MONTH_DRINKING].mean(axis=0)<lower_quartile_cutoff)[USERS] 
heavy_users_idx = np.where(data['complete record'][:,:,PAST_MONTH_DRINKING].mean(axis=0)>upper_quartile_cutoff)[USERS]

variable_filenames = [filename for filename in os.listdir(basepath) if 'initial-distribution' in filename]
demographics = {filename:np.loadtxt(make_filename(filename),delimiter=TAB) for filename in variable_filenames}
yvars = open('./agent-variables',READ).read().splitlines()
characteristics = ['initial-distribution-%s.txt'%('-'.join(yvar.split())) for yvar in yvars]

#-------MAIN
#Baseline demographics, Compare initial and final drinking distributions, Time series of consumption behavior
visualization.graph_everything(basepath=basepath,verbose=False,logfilename=logfilename)

#What is the evolution of the distribution of consumptive behavior?
visualization.snapshots(data['attitudes'],indices=[0,data['attitudes'].shape[1]/2-1,
	data['attitudes'].shape[1]-1],basepath=basepath,data_label='drinking behavior')

#Are the demographics of responders different?
if len(responders)>0:
	visualization.population_summary(basepath=basepath,criterion=list(responders),criterionname='responders')

#Are the demographics of the at-risk population different?
visualization.population_summary(basepath=basepath,criterion=map(int,data['at-risk']), criterionname='at risk')

#Are the dynamics of the pattern of consumption of the at-risk population different?
visualization.time_series(basepath=basepath,criterion=map(int,data['at-risk']),criterionname='at risk')

'''
	POSITIVE CONTROL: We hypothesize that those at-risk for drug consumption have different attitudes to the 
	medical consequences of consumption. Is this true?

	Compare general population | at-risk subpopulation | at-risk and responsive subpopulation
'''
visualization.plot_variable(data['attitudes'],basepath=basepath,
criterion=[list(responders),map(int,data['at-risk']),list(set(map(int,data['at-risk']))-set(responders))],
dataname='Intent to Use',criterionname=['Responders','At risk','Non-responders'])

'''
visualization.plot_variable(data['complete record'][:,:,PAST_MONTH_DRINKING].T,basepath=basepath,
criterion=[list(responders),map(int,data['at-risk']),list(set(map(int,data['at-risk']))-set(responders))],dataname='Drug Use',
criterionname=['Responders','At risk','Non-responders'])
#visualization.plot_variable(data['complete record'][:,:,PAST_MONTH_DRINKING])
'''

visualization.plot_variable(data['past month drinking'],basepath=basepath,
criterion=[list(responders),map(int,data['at-risk']),list(set(map(int,data['at-risk']))-set(responders))],dataname='Drug Use',
criterionname=['Responders','At risk','Non-responders'])
#visualization.plot_variable(data['complete record'][:,:,PAST_MONTH_DRINKING])


#Identify baseline characteristics of each quartile
#compare_demographics([('Heavy users',heavy_users_idx,'r'),('Light users',light_users_idx,'k')])

#compare_distributions('alpha',[('Heavy users',heavy_users_idx,'r'),('Light users',light_users_idx,'k')])

compare_distributions('alpha',[('Responders',list(responders),'r'),
								('Non-responders',list(set(range(data['complete record'].shape[1]))-set(responders)),'k')])
