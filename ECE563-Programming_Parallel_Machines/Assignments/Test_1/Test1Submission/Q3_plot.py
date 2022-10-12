import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys, os

# ticks delta
delta_x = 10
delta_t = 100

def plot(filename):
    base_name = os.path.basename(filename).split('.')[0]
    data = np.loadtxt(filename, delimiter=',')
    nT, nX = data.shape
    x = np.arange(nX)
    
    c = np.arange(1, nT+1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    fig =  plt.figure()
    plt.xticks(np.arange(0,nX+1,delta_x))
    plt.xlabel("X (rod length)")
    plt.ylabel("Temp")
    for t in range(nT):
        plt.plot(x, data[t,:], c=cmap.to_rgba(t))
    cbar = fig.colorbar(cmap, ticks=np.arange(0,nT+1,delta_t))
    cbar.set_label("Time Step", rotation=90)    
    plt.savefig(base_name+'.png')
    plt.close()
    print("plot saved for %s"%(filename))

if __name__ == "__main__":
    filename = sys.argv[1]
    assert os.path.exists(filename)
    plot(filename)