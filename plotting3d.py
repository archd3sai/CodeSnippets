import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(25,25))

for i in range(0,12):
    
    ax = fig.add_subplot(4, 3, i+1, projection='3d')
        
    y = range(200, 300)
    z = range(100, 200)
    x = range(0, 100)
    
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    ax.set_zlabel('zlabel')
    
    ax.plot3D(x, y, z, 'blue')
    ax.set_title('Figure ' + str(tub))

plt.show()
