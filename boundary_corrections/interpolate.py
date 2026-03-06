import numpy as np
import matplotlib.pyplot as plt


class interpolate_plots():
    def __init__(self):
        self.data_K1_K3=np.loadtxt('boundary_corrections/K1_K3plot.txt',skiprows=2)
        self.data_tau1=np.loadtxt('boundary_corrections/tau1.txt',skiprows=2)
        self.data_tau2=np.loadtxt('boundary_corrections/tau2.txt',skiprows=2)
        self.data_delta=np.loadtxt('boundary_corrections/delta.txt',skiprows=1)

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
        
    def get_tau2(self,x,lamda,k):
        "initialize lamda k and indices matrix"
        lamda_lst=np.array([0.35,0.5,0.707,1])
        k_lst=[[0.6],[0.4,0.5,0.6,0.7],[0.6],[0.4,0.5,0.6,0.7]]
        ind_lst=([[0],[1,1,1,2],[3],[4,4,5,6]])


        "calculate lamda positive"
        #search for closest lambda
        lamda_ana_lst=np.where(lamda_lst-lamda>=0,lamda_lst-lamda,1000)  #makes list only positive lamda differences
        lamda_ind_lst=np.where(np.round(lamda_ana_lst,6)==np.round(np.min(lamda_ana_lst),6))[0]  #searches for minimum

        #looks for if lamda is outside the regime and uses the closest value
        outr_cond=False
        outl_cond=False

        if lamda>lamda_lst[-1]:   #looks if higher than 1
            lamda_ind_lst[0]=lamda_ind_lst[-1]
            outr_cond=True

        elif lamda<lamda_lst[0]:   #looks if lower than 0.35
            outl_cond=True

        #use klst of certain lambda
        k_lst_new=np.array(k_lst[lamda_ind_lst[0]])
        #defines first lambda
        lamda_first=lamda_lst[lamda_ind_lst[0]]

        #interpolates k
        if len(k_lst_new)<2:   #looks for if there are no ks to interpolate
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][0]+1],x)
            y_first=y_geus  #uses the only k that there is
      
        elif k>k_lst_new[-1]:  #looks for if k is higher than max k
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][-1]+1],x)
            y_first=y_geus

        elif k<k_lst_new[0]:   #looks for if k is lower than min k
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][0]+1],x)
            y_first=y_geus

        else:  #interpolates k
            i=0
            y_geus_lst=[]
            for k_new in k_lst_new:  #cals y for all ks in the list and interpolates the closest k
                y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][i]+1],x)
                y_geus_lst.append(y_geus)
                i=i+1
            y_first=self.interpolate(k_lst_new,y_geus_lst,k)

        #returns if omega is out of bounds
        if outr_cond:
            return y_first
        if outl_cond:
            return y_first



        "calculates negative omega"
        #search for closest lambda
        lamda_ana_lst=abs(np.where(lamda_lst-lamda<0,lamda_lst-lamda,1000))
        lamda_ind_lst=np.where(np.round(lamda_ana_lst,6)==np.round(np.min(lamda_ana_lst),6))[0]

        #use klst of certain lambda
        k_lst_new=np.array(k_lst[lamda_ind_lst[0]])
        #defines first lambda
        lamda_second=lamda_lst[lamda_ind_lst[0]]

        #interpolates k
        if len(k_lst_new)<2:    #looks for if there are no ks to interpolate
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][0]+1],x)
            y_second=y_geus
           
        elif k>k_lst_new[-1]:   #looks for if k is higher than max k
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][-1]+1],x)
            y_second=y_geus

        elif k<k_lst_new[0]:    #looks for if k is lower than min k
            y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][0]+1],x)
            y_second=y_geus

        else:    #interpolates k
            i=0
            y_geus_lst=[]
            for k_new in k_lst_new:  #cals y for all ks in the list and interpolates the closest k
                y_geus=self.interpolate(self.data_tau2[:,0],self.data_tau2[:,ind_lst[lamda_ind_lst[0]][i]+1],x)
                y_geus_lst.append(y_geus)
                i=i+1
            y_second=self.interpolate(k_lst_new,y_geus_lst,k)

        "interpolates two ys for omegas"
        y=y_second+(y_first-y_second)/(lamda_first-lamda_second)*(lamda-lamda_second)
        return y
    
    def get_delta(self,x,lamda):
        lamda_lst=np.array([1,0.9,0.8,0.7,0.6,0.5])
        lamda_lst=np.array([0.5,0.6,0.7,0.8,0.9,1.0])

        y011_lst=np.zeros_like(lamda_lst)
        for i in range(6):
            y011_lst[len(lamda_lst)-i-1]=self.interpolate(self.data_delta[:,0],self.data_delta[:,i+1],x)

        y011=self.interpolate(lamda_lst,y011_lst,lamda)
        return y011

        


        
    
    

    
    
if __name__=='__main__':
    plots=interpolate_plots()
# data=np.loadtxt('boundary_corrections/delta.txt',skiprows=1)


# with open("boundary_corrections/delta.txt", "w") as f:
# #   for i in range(7):
#     f.write(f'x \t 1.0\t 0.9\t 0.8\t 0.7\t 0.6\t 0.5\n')
#     for i in range(10):
#         f.write(f'{data[i,0]} \t {data[i,1]}\t {data[i,3]}\t {data[i,5]}\t {data[i,6]}\t {data[i,4]}\t {data[i,2]}\n')


# data=np.loadtxt('boundary_corrections/K1_K3plot.txt',skiprows=2)
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

# data=np.loadtxt('boundary_corrections/tau1.txt',skiprows=2)
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

    
    # data=np.loadtxt('boundary_corrections/tau2.txt',skiprows=2)
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
    # lamda=0.4
    # k=0.54
    # x=0.12
    # ax.scatter(x,plots.get_tau2(x,lamda,k),label=r'$\lambda=$' f'{lamda}' r'$\,k=$' f'{k}')
    # # ax.scatter(data[:,0],data[:,7],label=f'K3')
    # # ax.vlines(0.11,0.9,1.05)
    # # ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
    # ax.legend()
    # ax.grid()
    # plt.show()


    data=np.loadtxt('boundary_corrections/delta.txt',skiprows=1)
    # y011=interpolate(data[:,0],data[:,2],0.11)
    # print(y011)
    plots=interpolate_plots()
    fig=plt.figure()
    ax=fig.subplots(1,1)
    ax.scatter(data[:,0],data[:,1],label=r'$\lambda=1.0$')
    ax.scatter(data[:,0],data[:,2],label=r'$\lambda=0.9$')
    ax.scatter(data[:,0],data[:,3],label=r'$\lambda=0.8$')
    ax.scatter(data[:,0],data[:,4],label=r'$\lambda=0.7$')
    ax.scatter(data[:,0],data[:,5],label=r'$\lambda=0.6$')
    ax.scatter(data[:,0],data[:,6],label=r'$\lambda=0.5$')
    lamda=0.94
    x=0.15
    ax.scatter(x,plots.get_delta(x,lamda),label=r'$\lambda=$' f'{lamda}' )


    # ax.scatter(data[:,0],data[:,7],label=f'K3')
    # ax.vlines(0.11,0.9,1.05)
    # ax.scatter(0.11,plots.get_K3(0.11),c='yellow')
    ax.legend()
    ax.grid()
    plt.show()


