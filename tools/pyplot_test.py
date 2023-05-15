# from matplotlib import pyplot, cm
# from numpy import linspace,add

# a = [ 5,3,6]
# b = [2,5,9]
# c = [ (255,255,255), (255,0,255), (255,255,0)]
# c = [ 'red', 'blue', 'green']

# x = linspace(-2,2, 1000)
# y = linspace(-2,2, 1000)

# z = add.outer(-10*x**2, -10*y**2)




# pyplot.scatter(x,y, c=z ,cmap=cm.viridis)
# pyplot.show()
import numpy as np;
import matplotlib as mpl;
import matplotlib.pyplot as plt;




figHandler = plt.figure();
cont_PSD = plt.contourf(plotFreq, plotCoord, plotPxx, 200, linestyle=None);


normi = mpl.colors.Normalize(vmin=-80, vmax=20);

colbar_PSD = plt.colorbar(cont_PSD);
colbar_PSD.set_norm(normi);
#colbar_PSD.norm = normi;
#mpl.colors.Normalize(vmin=-80, vmax=20);

plt.axis([1, 1000, -400, 400]);