import matplotlib.pyplot as plt
import utils.plotting
from utils.pickle import dump, load

results = load('data/di.sim.pickle')
returns = load('data/di.returns.pickle')

plt.figure(1)
for ret in returns.values():
    (xs,fs) = utils.plotting.cdf_points(ret)
    plt.plot(xs,fs)
plt.legend(returns.keys(),loc='best')

F = 2
for (name,result) in results.items():
    plt.figure(F)
    F += 1
    X = result.states
    A = result.actions
    C = result.costs
    (N,d,T) = X.shape
    plt.pcolor(C)
    plt.title('Cost image for ' + name)

plt.show()

