from hillcar import *
import mdp
import matplotlib.pyplot as plt
(MDP,G) = generate_hillcar_mdp(5,5)
(M,q) = MDP.tolcp()

plt.spy(M)
plt.show()

#v = mdp.value_iteration(MDP)
#mdp.plot_value(G,v)
