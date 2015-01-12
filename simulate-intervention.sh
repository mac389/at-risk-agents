mkdir -p $1

for ((count=10;count<20;count+=10)); do

	mkdir -p ${1}/${count}/targeted
	mkdir -p ${1}/${count}/random

	cp directory.json ${1}/${count}/targeted/directory.json
	cp directory.json ${1}/${count}/random/directory.json

	#Targeted
	python simulate-intervention.py -t ${1}/${count}/targeted -f $count
	python analyze-intervention.py -s ${1}/${count}/targeted

	#Random
	python simulate-intervention.py -t ${1}/${count}/random -r -f $count
	python analyze-intervention.py -s ${1}/${count}/random
done