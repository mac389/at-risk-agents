import os, json,re 

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt 
plt.switch_backend('Agg')

from matplotlib import rcParams
from optparse import OptionParser
from scipy.stats import percentileofscore,scoreatpercentile


rcParams['text.usetex'] = True
parser = OptionParser(usage="usage: %prog [options] filename",
                      version="%prog 1.0")
parser.add_option("-s", "--source",
                  action="store",
                  dest="source",
                  default=False,
                  help="Folder with data to analyze")
(options, args) = parser.parse_args()

READ = 'rb'
DELIMITER = '\t'
basepath = os.path.join(os.getcwd(),options.source)
hist_props={"range":[-1,1],"histtype":"stepfilled"}
make_filename = lambda filename: os.path.join(os.getcwd(),basepath,filename)
USERS = 0
TIME = 1
'''
    Questions to ask:
    1. Do those with the worst drinking behavior have a different alphas (susceptibilities) than those with the best
    2. Is targeted intervention (using our method of identification) effective
'''
verbose = False
data = {}
print 'ABM BASEPATH IS ',basepath
directory = json.load(open(os.path.join(os.getcwd(),basepath,'directory.json'),READ))
for variable in directory:
	if verbose:
		print 'Analyzing %s'%variable
	data[variable] = np.loadtxt(directory[variable],delimiter = DELIMITER)
	if variable == 'complete record':
		shape = tuple(map(int,re.findall('\d+',directory[variable]))[-3:])
		tmp = np.reshape(data[variable],shape) #-- Figure out shape from filename
		data[variable] = tmp

#Bounds for drinking behavior
upper_quartile_cutoff = scoreatpercentile(data['past month drinking'],75)
lower_quartile_cutoff = scoreatpercentile(data['past month drinking'],25)
light_users_idx = np.unique(np.where(data['past month drinking']<lower_quartile_cutoff)[USERS]) 

heavy_users_idx = np.where(data['past month drinking']>upper_quartile_cutoff)[USERS]

periods_of_intense_drinking = {agent:count for agent,count in enumerate(np.bincount(heavy_users_idx)) if count>0}
temporal_threshold = scoreatpercentile(periods_of_intense_drinking.values(),75)
heavy_frequent_users_idx = [agent for agent in periods_of_intense_drinking if periods_of_intense_drinking[agent] > temporal_threshold]
heavy_not_frequent_users_idx = np.array(list(set(periods_of_intense_drinking.keys()) - set(heavy_frequent_users_idx)))
heavy_users_idx = np.unique(heavy_users_idx)
#Identify baseline characteristics of each quartile
variable_filenames = [filename for filename in os.listdir(basepath) if 'initial-distribution' in filename]
demographics = {filename:np.loadtxt(make_filename(filename),delimiter=DELIMITER) for filename in variable_filenames}

nrows = 2
ncols = 3
normalize = lambda data: (data-data.min())/float(data.max()-data.min())

fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)
yvars = open('./agent-variables',READ).read().splitlines()
characteristics = ['initial-distribution-%s.txt'%('-'.join(yvar.split())) for yvar in yvars]

#Compare heavy users vs light users
for i,col in enumerate(axs):
	for j,row in enumerate(col):
			characteristic = characteristics[i*ncols+j]
			uq = demographics[characteristic][heavy_not_frequent_users_idx]
			lq = demographics[characteristic][heavy_frequent_users_idx]
					
			_,_,patches1=row.hist(uq,color='k',label=artist.format('Heavy Users'),range=(-1,1))
			plt.hold(True)
			_,_,patches2=row.hist(lq,color='r',alpha=0.5,label=artist.format('Heavy Frequent Users'),range=(-1,1))
			fig.canvas.mpl_connect('draw_event', artist.on_draw)
			artist.adjust_spines(row)
			if 'attitude' not in yvars[i*ncols+j]: 	
				row.set_xlabel(artist.format(yvars[i*ncols+j]))
			elif 'psychological' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude to','psychological','consequences']))
				row.set_xlabel(label)
			elif 'medical' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude','to medical','consequences']))
				row.set_xlabel(label)
plt.tight_layout()
fig.legend((patches1[0], patches2[0]), (artist.format('Heavy Users'),artist.format('Heavy Frequent Users')),
	loc='lower right', frameon=False, ncol=2)
#filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-no-temporal-threshold.png')
filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-frequent-vs-not-heavy.png')
plt.savefig(filename,dpi=300)

del fig,axs,i,j 
fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True)

#Compare heavy users vs frequent users
for i,col in enumerate(axs):
	for j,row in enumerate(col):
			characteristic = characteristics[i*ncols+j]
			uq = demographics[characteristic][heavy_users_idx]
			lq = demographics[characteristic][light_users_idx]
					
			_,_,patches1=row.hist(uq,color='k',label=artist.format('Heavy Users'),range=(-1,1))
			plt.hold(True)
			_,_,patches2=row.hist(lq,color='r',alpha=0.5,label=artist.format('Light Users'),range=(-1,1))
			fig.canvas.mpl_connect('draw_event', artist.on_draw)
			artist.adjust_spines(row)
			if 'attitude' not in yvars[i*ncols+j]: 	
				row.set_xlabel(artist.format(yvars[i*ncols+j]))
			elif 'psychological' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude to','psychological','consequences']))
				row.set_xlabel(label)
			elif 'medical' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude','to medical','consequences']))
				row.set_xlabel(label)
plt.tight_layout()
fig.legend((patches1[0], patches2[0]), (artist.format('Heavy Users'),artist.format('Light Users')),
	loc='lower right', frameon=False, ncol=2)
#filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-no-temporal-threshold.png')
filename = os.path.join(os.getcwd(),basepath,'compare-quartile-demographics-light-vs-heavy.png')
plt.savefig(filename,dpi=300)

del fig,axs 

data =  np.loadtxt(make_filename('alpha.txt'),delimiter=DELIMITER)
uq =data[heavy_users_idx]
lq = data[heavy_frequent_users_idx]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(uq,color='k',label=artist.format('Heavy Users'),range=(0,1),bins=20)
plt.hold(True)
ax.hist(lq,color='r',alpha=0.5,label=artist.format('Heavy Frequent Users'),range=(0,1),bins=20)
fig.canvas.mpl_connect('draw_event', artist.on_draw)
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('Prevalance'))
ax.set_xlabel(artist.format('Social Susceptibility'))
plt.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.05))
plt.tight_layout()
#plt.savefig(make_filename('susceptibility-no-temporal-threshold.png'),dpi=300)
plt.savefig(make_filename('susceptibility-frequent-vs-frequent-heavy.png'),dpi=300)



del fig,ax

data =  np.loadtxt(make_filename('alpha.txt'),delimiter=DELIMITER)
uq =data[heavy_not_frequent_users_idx]
lq = data[light_users_idx]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(uq,color='k',label=artist.format('Heavy Users'),range=(0,1),bins=20)
plt.hold(True)
ax.hist(lq,color='r',alpha=0.5,label=artist.format('Light Users'),range=(0,1),bins=20)
fig.canvas.mpl_connect('draw_event', artist.on_draw)
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('Prevalance'))
ax.set_xlabel(artist.format('Social Susceptibility'))
plt.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.05))
plt.tight_layout()
#plt.savefig(make_filename('susceptibility-no-temporal-threshold.png'),dpi=300)
plt.savefig(make_filename('susceptibility-heavy-light.png'),dpi=300)

#--- Create graphical output
visualization.graph_everything(basepath=basepath,moniker=options.target,verbose=False,logfilename=logfilename)
#Need a measure to show their behaviors are different
visualization.population_summary(moniker=options.target+'-at-risk',basepath=basepath,criterion=list(target_idx), criterionname='at risk')
visualization.time_series(moniker=options.target, basepath=basepath,criterion = list(target_idx),
	criterionname='at risk')

visualization.snapshots(drinking_behavior[:,start],drinking_behavior[:,stop-1],moniker='beta-%.02f'%beta,basepath=basepath)
