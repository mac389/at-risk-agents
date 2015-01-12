mkdir -p $1
cp directory.json ${1}/directory.json
python main-network-explore.py -t $1 -c conditions
python analyze-abm.py -s $1