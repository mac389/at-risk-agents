import json,random,os, logging, itertools, brewer2mpl
import matplotlib.pyplot as plt
import numpy as np 
import Graphics as artist

from scipy.stats import kruskal
from matplotlib import rcParams
from awesome_print import ap

rcParams['text.usetex'] = True
'''
 File Structure Summary 

    Variable : Filename

    initial_conditions : initial_conditions.txt 

'''

READ = 'rb'
TAB = '\t'
directory = json.load(open('directory.json',READ))
INITIAL = 0
END = -2

params = {
   'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True
   }

rcParams.update(params)

def norm(seqs):

	rng = np.ptp(np.array(seqs).ravel())
	mn = min(np.array(seqs).ravel())
	return map(lambda seq: 2*(seq-mn)/rng-1,seqs)

def perc(data):
   median = np.zeros(data.shape[1])
   perc_25 = np.zeros(data.shape[1])
   perc_75 = np.zeros(data.shape[1])
   for i in range(0, len(median)):
       median[i] = np.median(data[:, i])
       perc_25[i] = np.percentile(data[:, i], 25)
       perc_75[i] = np.percentile(data[:, i], 25)
   return median, perc_25, perc_75

def snapshots(data, indices,basepath=None, data_label='data'):
		indices = zip(indices,indices[1:])

		for start_idx,stop_idx in indices:
			initial_distribution = data[:,start_idx]
			final_distribution = data[:,stop_idx]

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.hist(initial_distribution,color='r',alpha=0.5,bins=20,label='Initial', range=(-1,1))
			ax.hist(final_distribution,color='k',alpha=0.5,bins=20,label='Final',range=(-1,1))
			artist.adjust_spines(ax)
			ax.set_xlabel(artist.format(data_label))
			ax.set_ylabel(artist.format('Prevalence'))

			H,p =kruskal(initial_distribution,final_distribution)
			effect_size = np.linalg.norm(final_distribution-initial_distribution)
			ax.annotate('\Large $d=%.02f, \; p=%.04f$'%(effect_size,p), xy=(.3, .9),  
				xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')
			plt.tight_layout()
			plt.legend(frameon=False)

			filename = os.path.join(basepath,'%s-compare-%d-%d.png'%(data_label,start_idx,stop_idx))
			plt.savefig(filename,dpi=300)	
			plt.close()

def graph_everything(basepath=None,verbose=False,logfilename=None):
	if logfilename:
		logging.basicConfig(filename=logfilename,level=logging.DEBUG)
	
	show_drinking_behavior(basepath)
	logging.info('Saved graph comparing initial and final distributions of drinking behavior')
	time_series(basepath)
	logging.info('Saved aggregate time series of drinking behavior')
	population_summary(basepath)
	logging.info('Saved heat map of population intent to drink')

def hist_compare(data,criterion=None, basepath=None, criterionname='Target population',fieldname='Field'):
	del fig,ax

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data,color='k',histtype='step',label=artist.format('Full Population'))
	plt.hold(True)
	if criterion:
		ax.hist(data[criterion],color='r',histtype='stepfilled',label=artist.format(criterionname))
	artist.adjust_spines(ax)
	ax.set_xlabel(artist.format(fieldname))
	ax.set_ylabel(artist.format('No. of people '))
	plt.legend(frameon=False)
	plt.tight_layout()
	plt.savefig(os.path.join(basepath,'hist_compare_full_%s'%('_'.join(criterion.split()))),dpi=300)

def show_drinking_behavior(basepath=None,compare_distributions=True,
	visualize_one_random_actor=False, visualize_all_actors=True):
	agents = np.loadtxt(os.path.join(basepath,'responders'),delimiter=TAB)
	filename = os.path.join(basepath,'drinking-behavior.txt')
	drinking_behavior = np.loadtxt(filename,delimiter=TAB)

	if compare_distributions:		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		H,p = kruskal(drinking_behavior[:,INITIAL],drinking_behavior[:,END])

		initial_distribution = drinking_behavior[:,INITIAL]
		final_distribution = drinking_behavior[:,END]

		low = min(initial_distribution.min(),final_distribution.min())
		high = max(initial_distribution.max(),final_distribution.max())

		ax.hist(initial_distribution,color='r',alpha=0.5,bins=20,label='Initial',range=(low,high))
		ax.hist(final_distribution,color='k',alpha=0.5,bins=20,label='Final', range=(low,high))
		artist.adjust_spines(ax)
		ax.set_xlabel(artist.format('Intent to drink'))
		ax.set_ylabel(artist.format('Prevalence'))
		plt.legend(frameon=False)
		filename = os.path.join(os.getcwd(),basepath,'drinking-behavior-compare-distributions.png')
		plt.savefig(filename,dpi=300)

	if visualize_one_random_actor:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		random_actor = random.choice(xrange(drinking_behavior.shape[0]))
		ax.plot(drinking_behavior[random_actor,:],'k--',linewidth=2)
		artist.adjust_spines(ax)
		ax.set_ylabel(artist.format('Past drinking behavior'))
		ax.set_xlabel(artist.format('Time'))
		filename = os.path.join(os.getcwd(),basepath,'drinking-behavior-visualize-actor.png')	
		plt.savefig(filename,dpi=300)

	if visualize_all_actors:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.imshow(drinking_behavior,interpolation='nearest',aspect='auto')
		artist.adjust_spines(ax)
		ax.set_ylabel(artist.format('Actor'))
		ax.set_xlabel(artist.format('Time'))
		plt.colorbar(cax)
		filename = os.path.join(os.getcwd(),basepath,'drinking-behavior-visualize-all-actors.png')		
		plt.savefig(filename,dpi=300)

def plot_variable(data,basepath=None,dataname='',criterion=None, criterionname=[]):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x = range(data.shape[1])
	ap('Plotting %s'%dataname)
	if criterion != None:
		if type(criterion) != list:
			median, lq, uq = perc(data[criterion,:])
			ax.plot(x,median,linewidth=2, color='#B22400')
			ax.fill_between(x, lq, uq, alpha=0.25, linewidth=0, color='#B22400')
		else:
			bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
			colors = bmap.mpl_colors
			for i,(x_criterion,x_label) in enumerate(itertools.izip_longest(criterion,criterionname,fillvalue='Group')):
				median, lq, uq = perc(data[x_criterion,:])
				ax.plot(x,median,linewidth=2, color=colors[i], label=artist.format(x_label))
				ax.fill_between(x, lq, uq, alpha=0.25, linewidth=0, color=colors[i])
	
	median, lq, uq = perc(data)
	ax.plot(x,median,linewidth=2, color='#B22400',label=artist.format('Full population'))
	ax.fill_between(x, lq, uq, alpha=0.25, linewidth=0, color='#B22400')
	
	artist.adjust_spines(ax)
	ax.set_ylabel(artist.format(dataname))
	ax.set_xlabel(artist.format('Time'))
	ax.axvline(data.shape[1]/3,color='r',linewidth=2,linestyle='--')
	ax.axvline(2*data.shape[1]/3,color='r',linewidth=2,linestyle='--')
	plt.legend(frameon=False,loc='lower left')
	plt.tight_layout()
	plt.savefig(os.path.join(basepath,'%s.png'%dataname))

def time_series(basepath=None, criterion=None, criterionname=''):
	filename = os.path.join(basepath,'attitudes.txt')
	attitudes = np.loadtxt(filename,delimiter=TAB)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#ax.fill_between(xrange(attitudes.shape[1]), attitudes.mean(axis=0)-attitudes.std(axis=0),
    #            attitudes.mean(axis=0) + attitudes.std(axis=0), color='k', alpha=0.4,
    #            label=artist.format('Full population'))
	ax.errorbar(xrange(attitudes.shape[1]),attitudes.mean(axis=0),yerr=(attitudes.std(axis=0)/attitudes.shape[0]))
#	ax.plot(xrange(attitudes.shape[1]),attitudes.mean(axis=0),color='k',linewidth=2)
	if criterion:
		data = attitudes[criterion]
		ax.fill_between(xrange(data.shape[1]), data.mean(axis=0)-data.std(axis=0),
                data.mean(axis=0) + data.std(axis=0), color='r', alpha=0.4,
                label=artist.format('criterionname'))
		ax.plot(xrange(data.shape[1]),data.mean(axis=0),color='r',linewidth=2)
	artist.adjust_spines(ax)
	ax.axvline(attitudes.shape[1]/3.,color='r',linewidth=2,linestyle='--') #This is a hack
	ax.axvline(2*attitudes.shape[1]/3.,color='r',linewidth=2,linestyle='--') #This is a hack
	ax.set_ylabel(artist.format('Intent to drink'))
	ax.set_xlabel(artist.format('Time'))
	ax.set_ylim(ymin=0)
	filename = os.path.join(os.getcwd(),basepath,'timecourse.png' if criterionname == '' else 'timecourse-%s.png'%criterionname)
	plt.savefig(filename,dpi=300)

def population_summary(basepath=None, criterion = None, criterionname=''):

	yvars = open(directory['variables'],READ).read().splitlines()
	yvars.remove('past month drinking')
	ncols = np.ceil(np.sqrt(len(yvars))).astype(int)
	nrows = np.ceil(len(yvars)/ncols).astype(int)
	MALE = 0.5
	FEMALE = 0.3
	fig,axs = plt.subplots(nrows=nrows,ncols=ncols,sharey=True)

	for i,col in enumerate(axs):
		for j,row in enumerate(col):
			filename = 'initial-distribution-%s.txt'%(yvars[i*ncols+j].replace(' ','-'))
			data = np.loadtxt(os.path.join(basepath,filename),delimiter=TAB)
			if criterion:
				weights = np.ones_like(data[criterion])/len(data[criterion])
				_,_,patches1 = axs[i,j].hist(data[criterion],color='r',alpha=0.5,
					label=artist.format(criterionname),histtype='step',weights=weights)
				plt.hold(True)
			weights = np.ones_like(data)/len(data)
			_,_,patches2 = axs[i,j].hist(data, color='k',label=artist.format('Full population'), 
				histtype='stepfilled' if not criterion else 'step',weights=weights)
			fig.canvas.mpl_connect('draw_event', artist.on_draw)
			artist.adjust_spines(axs[i,j])
			if 'attitude' not in yvars[i*ncols+j]: 	
				axs[i,j].set_xlabel(artist.format(yvars[i*ncols+j].replace('drink','use')))
				if 'gender' in yvars[i*ncols+j]:
					axs[i,j].set_xticks([FEMALE,MALE])
					axs[i,j].set_xticklabels(map(artist.format,['Female','Male']))
			elif 'psychological' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude to','psychological','consequences']))
				axs[i,j].set_xlabel(label)
			elif 'medical' in yvars[i*ncols+j]:
				label = '\n'.join(map(artist.format,['Attitude','to medical','consequences']))
				axs[i,j].set_xlabel(label)
				#axs[i,j].set_xlim([-50,50])

	plt.tight_layout()
	if criterion:
		fig.legend((patches1[0], patches2[0]), (artist.format(criterionname),artist.format('Full population')),
		loc='lower right', frameon=False, ncol=2)

	filename = os.path.join(os.getcwd(),basepath,'dashboard.png' if criterionname == '' else 'dashboard-%-s.png'%criterionname)
	plt.savefig(filename,dpi=300)
