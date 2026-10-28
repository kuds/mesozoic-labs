from math import comb
from fractions import Fraction
def hyp_le(k,a_tot,n1,N):  # P(X<=k), X successes in group of size n1, total successes a_tot, pop N
    return sum(comb(a_tot,x)*comb(N-a_tot,n1-x) for x in range(0,k+1))/comb(N,n1)
def hyp_ge(k,a_tot,n1,N):
    return sum(comb(a_tot,x)*comb(N-a_tot,n1-x) for x in range(k,min(a_tot,n1)+1))/comb(N,n1)
print('Fisher 2/3 vs 2/5 one-sided (r10 lower):', round(hyp_le(2,4,3,8),4))
print('Opt5 2/5 vs 2/5:', round(hyp_le(2,4,5,10),4), ' 0/5 vs 2/5:', round(hyp_le(0,2,5,10),4))
print('Opt1 4/4 vs 0/4:', round(hyp_ge(4,4,4,8),4),' 3/4 vs 0/4:', round(hyp_ge(3,3,4,8),4))
p=hyp_ge(3,3,4,8); print('stratified two strata 3/4 vs 0/4 each:', round(p*p,4))
print('RO 0/6 vs 4/6:',round(hyp_le(0,4,6,12),4),' 0/8 vs 5/8:',round(hyp_le(0,5,8,16),4),' 0/8 vs 4/8:',round(hyp_le(0,4,8,16),4), ' 0/6 vs 3/6:',round(hyp_le(0,3,6,12),4),' 0/7 vs 5/7', round(hyp_le(0,5,7,14),4))
# TABLE E via numeric integration: r10 ~ Beta(1+A, 1+n-A); r9 ~ Beta(3,4)
import numpy as np
from scipy import stats
a10=stats.beta(3,2); a9=stats.beta(3,4)
x=np.linspace(0,1,200001)
pdf9=a9.pdf(x); 
print('P(r10<r9)=',round(np.trapezoid(a10.cdf(x)*pdf9,x),4),' P(r10<r9/2)=',round(np.trapezoid(a10.cdf(x/2)*pdf9,x),4),' upper95=',round(a10.ppf(0.95),4))
print('Beta(3,4) 95% CI', round(a9.ppf(0.025),3), round(a9.ppf(0.975),3))
# beta-binomial P(2 of 3 | Beta(3,4))
from scipy.special import beta as B
print('P(2 of 3|unchanged)=', round(comb(3,2)*B(3+2,4+1)/B(3,4),4))
# entropy schedule
for t in (3145728,3932160,5505024,6000000,6004736,6291456,7000000):
    print('ent at',t, round(0.005*max(0,1-t/7e6),6))
print('t where bonus=0.0007:', 7e6*(1-0.0007/0.005))
