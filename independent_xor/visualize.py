import numpy as np 
import matplotlib.pyplot as plt 

sample = np.load( "new_xor_x.npy" )
label = np.load( "new_xor_y.npy" )

c = []
for i in range( label.shape[0] ):
    if label[i,0] == 1:
        c.append( "red" )
    else:
        c.append( "blue" )


fig, ax = plt.subplots()
ax.scatter(sample[:, 0], sample[:, 1], c=c, s=sample[:, 2]*10, alpha=0.8)

# ax.grid(True)
fig.tight_layout()
plt.savefig( "toy.png", pad_inches = 0.01, transparent = False )
plt.show()