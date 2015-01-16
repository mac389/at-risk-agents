import numpy as np
import copy 
from awesome_print import ap 
variable_filename = 'agent-variables'
READ = 'rb'

class actor(object):
	def __init__(self,baseline=None):
		self.MAX_ACTIVITY = 1
		#-------------------
		self.MALE_RISK = 0.5
		self.FEMALE_RISK = 0.3
		#---------Should really load these two from a parameter file
		self.update_rate = 0.01 #Can make this different for each agent
		self.filenames = open(variable_filename,READ).read().splitlines()
		if baseline == None:
			self.variables = {variable:False for variable in self.filenames}
			self.initialize()
		else:
			self.variables = baseline

	def internally_at_risk(self):
		return self.values > 0

	def initialize(self):
		initial_values = 2*np.random.random_sample(size=(len(self.variables),))-1
		for initial_value,key in zip(initial_values,self.variables.iterkeys()):
				self.variables[key] = initial_value 
		tmp = self.variables['gender'] 
		self.variables['gender'] = self.MALE_RISK if tmp >=0 else self.FEMALE_RISK
		self.values = self.variables['social hedonic values'] + self.variables['gender'] + self.variables['self-actualizing values']

	def social_sign(arr):
		return 1 if arr.sum()>0 else -1

	def calculate_intent_to_drink(self):
		PAST_MONTH_DRINKING = np.tanh(self.variables['past month drinking'])
		PSYCH_CONSEQUENCES = np.tanh(self.values+self.variables['attitude to psychological consequences'])
		
		MED_CONSEQUENCES = np.tanh(self.values+self.variables['attitude to medical consequences'])
		intent_to_drink = np.tanh(PAST_MONTH_DRINKING + PSYCH_CONSEQUENCES + MED_CONSEQUENCES)

		self.variables['past month drinking'] = intent_to_drink
		return intent_to_drink

	def update(self,update_dict):
		for variable in update_dict:
			tmp = self.variables[variable]
			self.variables[variable] = tmp + self.update_rate*update_dict[variable]
		tmp = self.calculate_intent_to_drink()
		self.variables['intent to drink'] = tmp

	def inspect_calculation(self): 
		#Assuing the output will be passed to a syntax highlighter like awesome print
		return 'Intent to drink: %.02f = Hx (%.02f) +  Medical (%.02f) + Psych(%.02f)'%(self.variables['past month drinking'],self.values,self.variables['attitude to medical consequences'],
			self.variables['attitude to psychological consequences'])

	def snapshot(self,as_dict = False,print_calc=False):
		if print_calc:
			print '\tValues = %.04f'%self.values
			print 'Med Chain = %04f'%(self.values+self.variables['attitude to medical consequences'])
			print 'Psych chain = %.04f'%(self.values+self.variables['attitude to psychological consequences'])
		return np.array([self.variables[variable] for variable in self.filenames]) if not as_dict else self.variables