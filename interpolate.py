import numpy as np
import matplotlib.pyplot as plt


class interpolate_plots():
    def __init__(self):
        self.data_K1_K3=np.loadtxt('K1_K3plot.txt',skiprows=2)
        self.data_tau1=np.loadtxt('tau1.txt',skiprows=2)

    def interpolate(self,x,y,x0):
        ind=np.where((x-x0)>0)[0][0]
        y0=y[ind-1]+(y[ind]-y[ind-1])/(x[ind]-x[ind-1])*(x0-x[ind-1])
        return y0 
    
    def get_K3(self,x):
        y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,5],x)
        return y011
    
    def get_K1(self,airfoil,x):
        if airfoil==66:
            y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,1],x)
            return y011
        
        if airfoil==65:
            y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,2],x)
            return y011
        
        if airfoil==64:
            y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,3],x)
            return y011
        
        if airfoil==4:
            y011=self.interpolate(self.data_K1_K3[:,0],self.data_K1_K3[:,4],x)
            return y011
        
    def get_tau1(self,x,b__h='circ'):
        if b__h=='circ':
            y011=self.interpolate(self.data_tau1[:,0],self.data_tau1[:,6],x)
            return y011

        else:
            # b__h=float(b__h)
            # print(b__h)
            B__H_lst=[2,1.75,1.5,1.43,1]
            B__H_lst=np.array([1,1.43,1.5,1.75,2])
            y011_lst=np.zeros(len(B__H_lst))
            for i in range(5):
                y011_lst[len(B__H_lst)-i-1]=self.interpolate(self.data_tau1[:,0],self.data_tau1[:,i+1],x)
            # print(B__H_lst)
            # print(y011_lst)
            y011=self.interpolate(B__H_lst,y011_lst,b__h)
            return y011
        
    def get_tau2(self):
        None

        

        
    
    

    
    

# data=np.loadtxt('delta.txt',skiprows=1)


# with open("newdelta.txt", "w") as f:
# #   for i in range(7):
#     f.write(f'x \t 1.0\t 0.9\t 0.8\t 0.7\t 0.6\t 0.5\n')
#     for i in range(10):
#         f.write(f'{data[i,0]} \t {data[i,1]}\t {data[i,3]}\t {data[i,5]}\t {data[i,6]}\t {data[i,4]}\t {data[i,2]}\n')


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
# ax.vlines(0.25,0.9,1.05)
# ax.scatter(0.25,plots.get_tau1(0.25,b__h=1.2),c='yellow')
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


# data=np.loadtxt('delta.txt',skiprows=1)
# # y011=interpolate(data[:,0],data[:,2],0.11)
# # print(y011)
# plots=interpolate_plots()
# fig=plt.figure()
# ax=fig.subplots(1,1)
# ax.scatter(data[:,0],data[:,1],label=r'$\lambda=1.0$')
# ax.scatter(data[:,0],data[:,2],label=r'$\lambda=0.5$')
# ax.scatter(data[:,0],data[:,3],label=r'$\lambda=0.9$')
# ax.scatter(data[:,0],data[:,4],label=r'$\lambda=0.6$')
# ax.scatter(data[:,0],data[:,5],label=r'$\lambda=0.8$')
# ax.scatter(data[:,0],data[:,6],label=r'$\lambda=0.7$')


# # ax.scatter(data[:,0],data[:,7],label=f'K3')
# # ax.vlines(0.11,0.9,1.05)
# # ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
# ax.legend()
# ax.grid()
# plt.show()


