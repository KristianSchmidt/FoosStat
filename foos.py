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


#printMat("Norm", normalized)

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

def oppositeSide(side):
    if side == 'b':
        return 'r'
    else:
        return 'b'

def oppositeSide5(side):
    if side == 'b5':
        return 'r5'
    else:
        return 'b5'

def pWinSet(probs, endscore, side, score_b, score_r, starting):
    if side = 'b' and score_b == endscore:
        return 1.0
    if side = 'r' and score_r == endscore:
        return 1.0
    if side = 'b' and score_r == endscore:
        return 0.0
    if side = 'r' and score_b == endscore:
        return 0.0

    opp = oppositeSide(side)
    oppStart = oppositeSide5(starting)

    pIfScore = pWinSet(probs, side, score_for + 1, score_against, oppStart)
    pScore = probs.loc[starting, side]
    

    pOppScores = probs.loc[starting, opp]
    pIfOppScores = pWinSet(probs, side, score_for, score_against + 1, starting)
    
    print("At score ", score_for, " - ", score_against)
    print("pScore: ", pScore)
    print("pIfScore: ", pIfScore)
    print("pOppScores: ", pOppScores)
    print("pIfOppScores: ", pIfOppScores)
    
    return  (pScore * pIfScore) + (pOppScores * pIfOppScores)

pWinSet(df, 'r', 3, 4, 'b5')



pWinSet(df, 'r', 0, 0, 'b5')
pWinSet(df, 'b', 0, 0, 'r5')
pWinSet(df, 'r', 0, 0, 'r5')

