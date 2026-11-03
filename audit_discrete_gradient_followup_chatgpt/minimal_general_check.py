from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'snapshot'))
from discrete_gradient import DiscreteGradient,v_to_ancestry_matrix
s=(R/'api_checks.py').read_text();exec(s[s.index('def adjacency'):s.index('rng=np.random')])
v=np.array([0,0,3],dtype=np.int64);D=np.ones((4,4));np.fill_diagonal(D,0);D[0,2]=D[2,0]=10
w=v.copy();w[2]=0
print('v=',v,'row=2 col=0','truth=',score(w,D)-score(v,D),'numba=',DiscreteGradient('numba').calculate_discrete_gradient(v,D)[2,0],'C=',DiscreteGradient('c').calculate_discrete_gradient(v,D)[2,0])
