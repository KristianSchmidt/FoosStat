import itertools
import numpy as np
import pandas

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

flatten = lambda l: [item for sublist in l for item in sublist]
states = ['b2', 'b3', 'b5', 'r2', 'r3', 'r5', 'g_b', 'g_r']

def indexInStates(s):
    return states.index(s)

dim = len(states)
transientDim = dim - 2

a = np.zeros(shape=(dim,dim))

with open('data.txt') as fp:
    lines = list(map((lambda x: x.rstrip('\n')), fp.readlines()))

lines = list(filter((lambda x: not x.startswith('set')), lines))

splits = [ x.split(',') for x in lines ]

windows = flatten([list(window(l,2)) for l in splits])

for event in windows:
    fromIdx = indexInStates(event[0])
    toIdx = indexInStates(event[1])
    #print(states, event[0], event[1], fromIdx,toIdx)
    a[(fromIdx,toIdx)] += 1

a[(indexInStates('g_b'), indexInStates('g_b'))] += 1
a[(indexInStates('g_r'), indexInStates('g_r'))] += 1

#print(a)

row_sums = a.sum(axis=1, keepdims=True)
normalized = a / row_sums

#print(normalized)

def printMat(name, mat):
    print(name, ": ", mat.shape)
    print(mat)


printMat("Norm", normalized)

Q = normalized[:transientDim, :transientDim]
#printMat("Q", Q)
It = np.identity(transientDim)
#printMat("It", It)
N = np.linalg.inv(It - Q)
#printMat("N", N)

R = normalized[0:transientDim, transientDim:]
#printMat("R", R)
B = np.matmul(N, R)
#printMat("B", B)

df = pandas.DataFrame(B)
df.columns = ['b', 'r']
df.index = states[:6]
print(df.loc[['b5', 'r5'],:])

def pRedWinSet(probs, endscore, score_r, score_b, starting):
    if score_r == endscore:
        return 1.0
    if score_b == endscore:
        return 0.0
    
    pIfScore = pRedWinSet(probs, endscore, score_r + 1, score_b, 'b5')
    pScore = probs.loc[starting, 'r']
    
    pIfNotScore = pRedWinSet(probs, endscore, score_r, score_b + 1, 'r5')
    pNotScore = probs.loc[starting, 'b']
    
    #print("At score ", score_r, " - ", score_b)
    #print("pScore: ", pScore)
    #print("pIfScore: ", pIfScore)
    #print("pOppScores: ", pIfNotScore)
    #print("pIfOppScores: ", pNotScore)
    
    return  (pScore * pIfScore) + (pNotScore * pIfNotScore)
    
pRedWinSet(df, 5, 0, 0, 'r5')

dfSet = pandas.DataFrame({'b': [1-pRedWinSet(df, 5, 0, 0, 'b5'), 1-pRedWinSet(df, 5, 0, 0, 'r5')], 'r': [pRedWinSet(df, 5, 0, 0, 'b5'), pRedWinSet(df, 5, 0, 0, 'r5')]}, index=['b5','r5'])

pRedWinSet(dfSet, 3, 0, 0, 'r5')

