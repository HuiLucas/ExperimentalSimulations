import numpy as np
import matplotlib.pyplot as plt


class interpolate_plots():
    def __init__(self):
        self.data_K1_K3=np.loadtxt('K1_K3plot.txt',skiprows=2)

    def interpolate(self,x,y,x0):
        ind=np.where((x-x0)>0)[0][0]
        y0=y[ind-1]+(y[ind]-y[ind-1])/(x[ind]-x[ind-1])*(x0-x[ind-1])
        return y0 
    
    def get_K3(self,x):
        y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,2],x)
        return y011
    
    




# data=np.loadtxt('K1_K3plot.txt',skiprows=2)
# # y011=interpolate(data[:,0],data[:,2],0.11)
# # print(y011)
# plots=interpolate_plots()
# fig=plt.figure()
# ax=fig.subplots(1,1)
# ax.scatter(data[:,0],data[:,1],label=f'K1 66')
# ax.scatter(data[:,0],data[:,2],label=f'K1 65')
# ax.scatter(data[:,0],data[:,3],label=f'K1 64')
# ax.scatter(data[:,0],data[:,4],label=f'K1 4')
# ax.scatter(data[:,0],data[:,5],label=f'K3')
# # ax.scatter(data[:,0],data[:,6],label=f'K3')
# # ax.scatter(data[:,0],data[:,7],label=f'K3')
# ax.vlines(0.11,0.9,1.05)
# ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
# ax.legend()
# ax.grid()
# plt.show()

# data=np.loadtxt('tau1.txt',skiprows=2)
# # y011=interpolate(data[:,0],data[:,2],0.11)
# # print(y011)
# plots=interpolate_plots()
# fig=plt.figure()
# ax=fig.subplots(1,1)
# ax.scatter(data[:,0],data[:,1],label=f'2')
# ax.scatter(data[:,0],data[:,2],label=f'1.75')
# ax.scatter(data[:,0],data[:,3],label=f'1.5')
# ax.scatter(data[:,0],data[:,4],label=f'1.43')
# ax.scatter(data[:,0],data[:,5],label=f'1')
# ax.scatter(data[:,0],data[:,6],label=f'circ')
# # ax.scatter(data[:,0],data[:,7],label=f'K3')
# # ax.vlines(0.11,0.9,1.05)
# # ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
# ax.legend()
# ax.grid()
# plt.show()


# data=np.loadtxt('tau2.txt',skiprows=2)
# # y011=interpolate(data[:,0],data[:,2],0.11)
# # print(y011)
# plots=interpolate_plots()
# fig=plt.figure()
# ax=fig.subplots(1,1)
# ax.scatter(data[:,0],data[:,1],label=r'$\lambda=0.35\, k=0.6$')
# ax.scatter(data[:,0],data[:,2],label=r'$\lambda=0.5\, k=0.4,0.5,0.6$')
# ax.scatter(data[:,0],data[:,3],label=r'$\lambda=0.5\, k=0.7$')
# ax.scatter(data[:,0],data[:,4],label=r'$\lambda=0.707\, k=0.6$')
# ax.scatter(data[:,0],data[:,5],label=r'$\lambda=1.0 \,k=0.4,0.5$')
# ax.scatter(data[:,0],data[:,6],label=r'$\lambda=1.0 \,k=0.6$')
# ax.scatter(data[:,0],data[:,7],label=r'$\lambda=1.0 \,k=0.7$')

# # ax.scatter(data[:,0],data[:,7],label=f'K3')
# # ax.vlines(0.11,0.9,1.05)
# # ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
# ax.legend()
# ax.grid()
# plt.show()


data=np.loadtxt('delta.txt',skiprows=1)
# y011=interpolate(data[:,0],data[:,2],0.11)
# print(y011)
plots=interpolate_plots()
fig=plt.figure()
ax=fig.subplots(1,1)
ax.scatter(data[:,0],data[:,1],label=r'$\lambda=0.1$')
ax.scatter(data[:,0],data[:,2],label=r'$\lambda=0.5$')
ax.scatter(data[:,0],data[:,3],label=r'$\lambda=0.9$')
ax.scatter(data[:,0],data[:,4],label=r'$\lambda=0.6$')
ax.scatter(data[:,0],data[:,5],label=r'$\lambda=0.8$')
ax.scatter(data[:,0],data[:,6],label=r'$\lambda=0.7$')


# ax.scatter(data[:,0],data[:,7],label=f'K3')
# ax.vlines(0.11,0.9,1.05)
# ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
ax.legend()
ax.grid()
plt.show()


