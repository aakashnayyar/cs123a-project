import numpy as np
import pandas as pd

# create state space and initial state probabilities

hidden_states = ['N', 'C']
pi = [1, 0]
print('\n')
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n')

# create hidden transition matrix
# a or alpha 
#   = transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.7, 0.3]  #need to update
a_df.loc[hidden_states[1]] = [0.4, 0.6]  #need to update

print(a_df)

a = a_df.values
print('\n')


# create matrix of observation (emission) probabilities
# b or beta = observation probabilities given state
# matrix is size (M x O) where M is number of states 
# and O is number of different possible observations

states = ['A', 'T', 'C', 'G']
observable_states = states

b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [0.31, 0.19, 0.19, 0.31]  # added 0.1 to T
b_df.loc[hidden_states[1]] = [0.2, 0.1, 0.5, 0.2]      #need to update

print(b_df)

b = b_df.values
print('\n')

# observation sequence of dog's behaviors
# observations are encoded numerically

obs_map = {'A':0, 'T':1, 'C':2, 'G':3}

fn = "example.fna.txt"
with open(fn) as f:
    #acc = f.readline()  #first line is accession number if fasta file
    seq = f.read()
    seq = "".join(seq.split("\n"))
    
#s = 'AAATGGTATCC'
mapped = [obs_map[x] for x in seq] 
obs = np.array(mapped)             # mapped seq of nts

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

def viterbi(pi,a,b,obs):

	nStates = np.shape(b)[0]
	T = np.shape(obs)[0]

	path = np.zeros(T)
	delta = np.zeros((nStates,T))
	phi = np.zeros((nStates,T))

	delta[:,0] = pi * b[:,obs[0]]
	phi[:,0] = 0

	for t in range(1,T):
		for s in range(nStates):
			delta[s,t] = np.max(delta[:,t-1]*a[:,s])*b[s,obs[t]]
			phi[s,t] = np.argmax(delta[:,t-1]*a[:,s])

	path[T-1] = np.argmax(delta[:,T-1])
	for t in range(T-2,-1,-1):
		#path[t] = phi[int(path[t+1]): int(t+1) , int(t+1)]
		path[t] = phi[int(path[t+1]) , int(t+1)]

	return path,delta, phi

path, delta, phi = viterbi(pi, a, b, obs)

state_map = {0:'N', 1:'C'}
state_path = [state_map[v] for v in path]

print('best path: ', ''.join(state_path))
