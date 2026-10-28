# Recheck: independent exact computations (Fisher one-sided p, power, beta-binomial, schedule)
from math import comb
def fisher_greater(a, n1, b, n2):
    # P(X >= a) where X = alarms in arm1 given total a+b, hypergeometric
    K = a + b; N = n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)
def fisher_less(a, n1, b, n2):
    K = a + b; N = n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(max(0, K - n2), a + 1)) / comb(N, n1)
def binom(k, n, p): return comb(n, k) * p**k * (1-p)**(n-k)
def power(n, p1, p2, alpha=0.05):
    s = 0.0
    for a in range(n+1):
        for b in range(n+1):
            if fisher_greater(a, n, b, n) <= alpha:
                s += binom(a, n, p1) * binom(b, n, p2)
    return s
print("Fisher r10 2/3 vs r9 2/5 (less):", round(fisher_less(2,3,2,5),4))
print("Fisher 2/5 vs 2/5:", round(fisher_less(2,5,2,5),4), " 0/5 vs 2/5:", round(fisher_less(0,5,2,5),4))
print("0/3 vs 2/5:", round(fisher_less(0,3,2,5),4), "0/2:", round(fisher_less(0,2,2,5),4), "0/1:", round(fisher_less(0,1,2,5),4))
print("4/6 vs 0/6:", round(fisher_greater(4,6,0,6),4), "5/8 vs 0/8:", round(fisher_greater(5,8,0,8),4), "4/8 vs 0/8:", round(fisher_greater(4,8,0,8),4))
print("4/4 vs 0/4:", round(fisher_greater(4,4,0,4),4), "3/4 vs 0/4:", round(fisher_greater(3,4,0,4),4))
# stratified: sum over two strata of X, each hypergeom N=8,K=3,n=4; P(S>=6)
from itertools import product
def hyp(N,K,n):
    return {x: comb(K,x)*comb(N-K,n-x)/comb(N,n) for x in range(max(0,n-(N-K)), min(K,n)+1)}
h = hyp(8,3,4)
print("stratified 3/4 v 0/4 x2:", round(sum(h[x]*h[y] for x in h for y in h if x+y>=6),4))
for n in (6,8):
    print("n",n,"power 2/3 v 0:",round(power(n,2/3,0),3)," 2/3 v 1/3:",round(power(n,2/3,1/3),3)," 0.4 v 0:",round(power(n,0.4,0),3))
# beta-binomial Beta(3,4), n=3
from math import gamma
def B(a,b): return gamma(a)*gamma(b)/gamma(a+b)
print("betabinom Beta(3,4) n=3:", [round(comb(3,k)*B(3+k,4+3-k)/B(3,4),3) for k in range(4)])
# uniform posterior Beta(3,2) 95% quantile and exact binomial upper bound for 2/3
lo,hi=0,1
for _ in range(60):
    m=(lo+hi)/2
    F=4*m**3-3*m**4
    lo,hi=(m,hi) if F<0.95 else (lo,m)
print("Beta(3,2) q95:", round(lo,4), " exact 2/3 upper:", round(0.95**(1/3),4))
# P(r10 rate < r9 rate), r10 ~ Beta(3,2) (uniform prior, 2/3), r9 ~ Beta(3,4); numeric
import random
N=400000; random.seed(1)
c=sum(random.betavariate(3,2)<random.betavariate(3,4) for _ in range(N))/N
c2=sum(random.betavariate(3,2)<0.5*random.betavariate(3,4) for _ in range(N))/N
print("MC P(r10<r9):", round(c,3), " P(r10<half):", round(c2,3))
# schedule
for s in (1048576,3145728,3932160,5505024,6000000,6021120,6291456):
    print("ent_coef at",s, round(0.005*max(0,1-s/7e6),6))
