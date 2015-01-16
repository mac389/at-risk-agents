#This is incorrect, this is just targeting the most connected people. 
number_of_concerning_influencers_per_agent = {agent:sum([G.node[predecessor]['actor'].internally_at_risk() 
														for predecessor in G.predecessors(agent)]) 
														for agent in G.nodes()}

threshold_for_number_of_concerning_influencers = scoreatpercentile(number_of_concerning_influencers_per_agent.values(),75)

susceptible_agents_receiving_too_much_negative_influence = [agent for agent in susceptible_agents 
					 if sum([G.node[influencer]['actor'].internally_at_risk() 
 					for influencer in G.predecessors(agent)]) > threshold_for_number_of_concerning_influencers]

distribution_of_increases_in_drinking =  np.diff(drinking_behavior,axis=1)
distribution_of_increases_in_drinking = distribution_of_increases_in_drinking[distribution_of_increases_in_drinking>0]
threshold_for_concerning_hx_of_drinking = scoreatpercentile(distribution_of_increases_in_drinking,75)

recent_drinking_history = {agent:hx for agent,hx in enumerate(np.diff(drinking_behavior[:,:start],axis=1).mean(axis=1))}

threshold_for_permissive_attitudes = scoreatpercentile([G.node[agent]['actor'].variables['attitude to medical consequences'] for agent in G.nodes()],75)

'''
susceptible_agents_receiving_too_much_negative_influence_and_starting_to_consume = [agent for agent 
		in susceptible_agents_receiving_too_much_negative_influence if 
		recent_drinking_history[agent]>threshold_for_concerning_hx_of_drinking
		and G.node[agent]['actor'].variables['attitude to medical consequences'] > threshold_for_permissive_attitudes]
'''

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


	'''
	_,baseline,_ = np.array_split(data['complete record'][:,list(set(set(range(data['attitudes'].shape[0]))-set(data['at-risk']))),:].mean(axis=0),3)
	responders = [agent for agent in data['at-risk'] if responded(agent,baseline)]
	zero_crossings = [np.where(np.diff(np.sign(row)))[0] for row in data['complete record'][:,:,PAST_MONTH_DRINKING]]
	zero_crossings = [(i,row) for i,row in enumerate(zero_crossings) if len(row)>0]
	agents,crossing_idx = zip(*zero_crossings)
	'''

	if bubble_plot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		#Maybe I should use the non-parametric equivalents
		for condition, color,label in zip(np.split(drinking_behavior,2,axis=1),['k','r'],['Baseline','Intervention']):
			mu = condition.mean(axis=0)
			sigma = condition.std(axis=0)
			ax.plot(mu,sigma,'%s.'%color,label=artist.format(label))
			plt.hold(True)
		artist.adjust_spines(ax)
		ax.set_xlabel(r'\Large Average intent to drink $\left(\mu\right)$')
		ax.set_ylabel(r'\Large Deviation in intent to drink $\left(\sigma\right)$')
		filename = os.path.join(os.getcwd(),basepath,'drinking-behavior-bubble.png')
		plt.savefig(filename,dpi=300)

del fig,ax 

		fig = plt.figure()
		ax = fig.add_subplot(111)
		for condition,color,label in zip(np.split(drinking_behavior,2,axis=1),['k','r'],['Baseline', 'Intervention']):
				mu = condition[list(agents),:].mean(axis=0)
				sigma = condition[list(agents),:].std(axis=0)
				ax.plot(mu,sigma,'%s.'%color,label=artist.format(label))
				plt.hold(True)
		artist.adjust_spines(ax)
		ax.set_xlabel(r'\Large Average intent to drink $\left(\mu\right)$')
		ax.set_ylabel(r'\Large Deviation in intent to drink $\left(\sigma\right)$')
		filename = os.path.join(os.getcwd(),basepath,'drinking-behavior-bubble-responders.png')
		plt.savefig(filename,dpi=300)