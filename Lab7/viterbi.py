#code for viterbi algorithm
"""
params is a triple (pi, A, B) where
pi = initial probability distribution over states
A = transition probability matrix
B = emission probability matrix

observations is the sequence of observations (in our case, the observed words)

the function returns the optimal sequence of states and its score
"""
def viterbi(params, observations):
    pi, A, B = params
    M = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf') #cases that have not been treated
    backpointers = np.zeros((M, S), 'int')

    # base case
    alpha[0, :] = pi * B[:,observations[0]]

    # recursive case
    for t in range(1, M):
        for s2 in range(S):
            for s1 in range(S):
                score = alpha[t-1, s1] * A[s1, s2] * B[s2, observations[t]]
                if score > alpha[t, s2]:
                    alpha[t, s2] = score
                    backpointers[t, s2] = s1
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])

    return list(reversed(ss)), np.max(alpha[M-1,:])
