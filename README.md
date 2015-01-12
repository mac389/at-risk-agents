at-risk-agents
==============

Identifying at-risk individuals in social networks


To run a simulation type _./simulate-intervention.sh NAME_. (It is an executable.) This creates a folder called _NAME_ with subfolders corresponding to the maximum fraction of users being targeted. This fraction can be changed in _simulate-itervention.sh_. 


Two subsubfolders are created in each subfolder, _targeted_ and _random_. _Targeted_ contains the results of the simulation targeted at-risk agents. _Random_ contains the results of the simulation targeting a group of agents of the same size at the at-risk group chosen from the general population, including the at-risk subpopulation. 

After the simulations _simulate-intervention.sh adds the directories where the data were stored to .gitignore. 

Technical notes
================

 1. A record of each simulation is saved as a tab-delimited text-file with four significant figures. For 200 time steps and 100,000 agents, this file is approximately 1 GB. Saving it in binary or as a 4 digit integer will save significant space. 

 1. This version uses forward Euler integration with a times step that is 1% of the time constant. This may be numerically unstable, increasingly so for longer simulations. 

Dependencies
================

 1. NumPy
 1. SciPy
 1. Matplotlib
 1. Awesome Print
 1. NetworkX
 1. ProgressBar
 1. Termcolor 