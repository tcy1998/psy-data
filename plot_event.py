import pandas as pd
import csv
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# from sklearn.preprocessing  import PolynomialFeatures

F_NAME = 'model_params_subject_reduced_2018_02_15_14_17_58.pkl'
F_NAME1 = 'test_data_event.pkl'
F_NAME2 = 'train_data_event.pkl'


# class F_beta(object):
#     def __init__(self, model_param):
#         ######################################################
#         ### The model param is directly from the pkl file. ###
#         ######################################################
#         Poly_deg = 3
#         self.beta = model_param['learned']['beta'].flatten()
#         self.poly = PolynomialFeatures(degree=Poly_deg, include_bias=True)
#     def y_hat(self, X):
#         X_ = self.poly.fit_transform(X.reshape(1,-1))
#         return (np.dot(X_.flatten(), self.beta)).flatten()

data = pd.read_pickle(F_NAME2)
# print(data)

x0, y0, z0 = [], [], []
vx, vy, vz = [], [], []
dist, vel = [], []
v1, v2, v3 = [], [], []

def fun(x): #处理符号问题
    round(x,2)
    if x>=0: return '+'+str(x)
    else: return str(x)

for d in data:
    x = d['X']
    Z = d['y']
    subject = d['label']
    # print(subject)
    # print(Y)
    v1.append(Z)
    # v2.append(Y_pred)
    # v3.append(Y_pred2)

    x0.append(x[0])
    y0.append(x[1])
    z0.append(x[2])

    vx.append(x[4])
    vy.append(x[5])
    vz.append(x[6])

    dist.append(x[3])
    vel.append(x[7])


    # plt.savefig('scatter_plot/'+subject[0]+' '+subject[1]+'dis_vel.png')
    # plt.show()

    # plt.scatter(x[7], Z, c='b')
    plt.show()
    X = x[3]
    Y = x[7]
    print(Y)
    # print(X)
    n = len(X)
    sigma_x = 0
    for i in X : sigma_x+=i
    sigma_y = 0
    for i in Y : sigma_y+=i
    sigma_z = 0
    for i in Z : sigma_z+=i
    sigma_x2 = 0
    for i in X : sigma_x2+=i*i
    sigma_y2 = 0
    for i in Y : sigma_y2+=i*i
    sigma_x3 = 0
    for i in X : sigma_x3+=i*i*i
    sigma_y3 = 0
    for i in Y : sigma_y3+=i*i*i
    sigma_x4 = 0
    for i in X : sigma_x4+=i*i*i*i
    sigma_y4 = 0
    for i in Y : sigma_y4+=i*i*i*i
    sigma_x_y = 0
    for i in range(n) : 
        sigma_x_y+=X[i]*Y[i]
    #print(sigma_xy)
    sigma_x_y2 = 0
    for i in range(n) : sigma_x_y2+=X[i]*Y[i]*Y[i]
    sigma_x_y3 = 0
    for i in range(n) : sigma_x_y3+=X[i]*Y[i]*Y[i]*Y[i]
    sigma_x2_y = 0
    for i in range(n) : sigma_x2_y+=X[i]*X[i]*Y[i]
    sigma_x2_y2 = 0
    for i in range(n) : sigma_x2_y2+=X[i]*X[i]*Y[i]*Y[i]
    sigma_x3_y = 0
    for i in range(n) : sigma_x3_y+=X[i]*X[i]*X[i]*Y[i]
    sigma_z_x2 = 0
    for i in range(n) : sigma_z_x2+=Z[i]*X[i]*X[i]
    sigma_z_y2 = 0
    for i in range(n) : sigma_z_y2+=Z[i]*Y[i]*Y[i]
    sigma_z_x_y = 0
    for i in range(n) : sigma_z_x_y+=Z[i]*X[i]*Y[i]
    sigma_z_x = 0
    for i in range(n) : sigma_z_x+=Z[i]*X[i]
    sigma_z_y = 0
    for i in range(n) : sigma_z_y+=Z[i]*Y[i]

    a=np.array([[sigma_x4,sigma_x3_y,sigma_x2_y2,sigma_x3,sigma_x2_y,sigma_x2],
               [sigma_x3_y,sigma_x2_y2,sigma_x_y3,sigma_x2_y,sigma_x_y2,sigma_x_y],
               [sigma_x2_y2,sigma_x_y3,sigma_y4,sigma_x_y2,sigma_y3,sigma_y2],
               [sigma_x3,sigma_x2_y,sigma_x_y2,sigma_x2,sigma_x_y,sigma_x],
               [sigma_x2_y,sigma_x_y2,sigma_y3,sigma_x_y,sigma_y2,sigma_y],
               [sigma_x2,sigma_x_y,sigma_y2,sigma_x,sigma_y,n]])
    b=np.array([sigma_z_x2,sigma_z_x_y,sigma_z_y2,sigma_z_x,sigma_z_y,sigma_z])

    res= np.linalg.solve(a,b)
    print("z=%.6s*x^2%.6s*xy%.6s*y^2%.6s*x%.6s*y%.6s"%(fun(res[0]),fun(res[1]),fun(res[2]),fun(res[3]),fun(res[4]),fun(res[5])))

    fig = plt.figure()#建立一个空间
    ax =fig.add_subplot(111,projection='3d')# 3D坐标
 
    m = 100
    u1 = np.linspace(0,100,m)
    u2 = np.linspace(-50,50,m)
    x1,y1 = np.meshgrid(u1,u2) #转化成矩阵
 
    #给出方程
    z1=res[0]*x1*x1+res[1]*x1*y1+res[2]*y1*y1+res[3]*x1+res[4]*y1+res[5]
    #画出曲面
    ax.plot_surface(x1,y1,z1,rstride=3,cstride=3,cmap=cm.jet)
    # ax = plt.subplot(111, projection='3d')
    ax.scatter(x[3], x[7], Z, c='y')
    ax.set_xlabel('distance')
    ax.set_ylabel('velocity')
    ax.set_zlabel('variable')
    plt.savefig('event_surface/'+subject[0]+' '+subject[1]+'dis_vel.png')
    # plt.show()