import numpy as np
import pandas as pd

def viterbi(pi,a,b,obs):
    """ This algorithm finds the most probable sequence of states.
        It determines the most likely sequence of hidden states (labels designating
        coding(C) or non-coding(N) region) based on the whole observed DNA sequence.  
    """

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
        path[t] = phi[int(path[t+1]) , int(t+1)]

    return path,delta, phi


def get_strand_info(states):
    """ Returns information about predicted Genes from strand.
    """
    
    # lists to collect info
    gene_num = []
    left_end = []
    right_end = []
    gene_len = []
    # tracks current gene number
    cur_gene_num = 0
    gene_count = 0

    # iterates through whole sequence
    for i in range(len(states)):
        # detects when a coding region is reached
        if states[i] == 'C':
            # checks if coding nucleotide is at beginning
            if i-1 == -1 or states[i-1] == 'N':
                cur_gene_num = cur_gene_num + 1
                gene_num.append(cur_gene_num)
                left_end.append(i+1)
            # checks if coding nucleotide is at end
            if i+1 == len(states) or states[i+1] == 'N':
                right_end.append(i+1)
                gene_len.append(right_end[cur_gene_num-1]-left_end[cur_gene_num-1] + 1)

    # collects data into data frame and returns results
    results = {
        'Gene': gene_num,
        'LeftEnd': left_end,
        'RightEnd': right_end,
        'GeneLength': gene_len
    }
    return pd.DataFrame(results)


# MAIN

# create state space and initial state probabilities
hidden_states = ['N', 'C']  
pi = [0.5, 0.5]

print('\n')
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
print('\n')

# create hidden transition matrix
# a or alpha 8= transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.996, 0.004]         
a_df.loc[hidden_states[1]] = [0.002, 0.998]

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
b_df.loc[hidden_states[0]] = [0.32, 0.32, 0.18, 0.18] 
b_df.loc[hidden_states[1]] = [0.28, 0.25, 0.22, 0.25]

print(b_df)
b = b_df.values
print('\n')

# observation nucleotides are encoded numerically
obs_map = {'A':0, 'T':1, 'C':2, 'G':3}

fn = ["seq1.txt", "seq2.txt", "seq3.txt"]  #fasta files we are using
for file in fn:
    with open(file) as f:
        acc = f.readline()  #first line is accession number in fasta file
        seq = f.read()
        seq = "".join(seq.split("\n"))
    
    mapped = [obs_map[x] for x in seq] 
    obs = np.array(mapped)             # mapped sequence of nucleotides

    inv_obs_map = dict((v,k) for k, v in obs_map.items())
    obs_seq = [inv_obs_map[v] for v in list(obs)]

    # call viterbi function to get path of hidden states
    path, delta, phi = viterbi(pi, a, b, obs)  

    state_map = {0:'N', 1:'C'}
    state_path = [state_map[v] for v in path]  #mapped state path


    # prints strand info for given sequence
    print('\n')
    print(file + ':')
    print(get_strand_info(state_path).to_string(index=False))

    #print best path of hidden states
    print('best path: ', ''.join(state_path))
