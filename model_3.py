import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma,gammaln
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from cv1_image_processing import plot_ROC_curve
from scipy.optimize import fminbound
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


train_size = 1000
test_size = 100

class students_tDist():
    
    def __init__(self,mu,covariance,v):
        self.mean = mu
        self.covariance = covariance
        self.v = v
        self.E_h = np.zeros(train_size)
        self.E_log_h = np.zeros(train_size)
        self.delta = np.zeros(train_size)
    
    def prob(self,i_index,X):
        D = self.mean.shape[0]
        val1 = gamma((self.v + D)/2) / ( ((self.v * np.pi)** D/2) *np.sqrt(det(self.covariance))*gamma(self.v/2) )
        term = np.matmul( (X[:,i_index].reshape(-1,1)-self.mean).T,inv(self.covariance) )                                  
        delta = np.matmul(term,(X[:,i_index].reshape(-1,1) - self.mean))
        val2 = (1 + delta/self.v)
        val = val1 * pow(val2, -(self.v+D)/2)
        return val[0,0]
    
    def argmin_v(self):
        v = fminbound(t_cost, 0, 10, args=(self.E_h, self.E_log_h)) 
        return v
    
    def apply_EM(self,X):
        D = self.mean.shape[0]
        
        #expectation step
        print("Expecting")
        for i in range(0,train_size):
            term = np.matmul( (X[:,i].reshape(-1,1)-self.mean).T , inv(self.covariance) )
            delta = np.matmul(term , (X[:,i].reshape(-1,1) - self.mean))                                  
            self.delta[i] = delta
            self.E_h[i] = (self.v+D)/(self.v + delta)
            self.E_log_h[i] = digamma((self.v+D)/2) - np.log((self.v+delta)/2)
        
        #maximization step
        print("Maximizing") 
        
        #mean update
        self.mean = (np.sum(self.E_h * X, axis=1)/np.sum(self.E_h)).reshape(D,1)
        
        #covariance update
        num = np.zeros((D,D))
        for i in range(0,train_size):
            prod = np.matmul((X[:,i].reshape(-1,1) - self.mean), (X[:,i].reshape(-1,1) - self.mean).T)
            num = num + self.E_h[i]*prod
        self.covariance = num/np.sum(self.E_h)
        self.covariance = np.diag( np.diag(self.covariance) )
        
        #updating dof via argmin
        self.v = self.argmin_v()
 
        for i in range(0,train_size):
            term = np.matmul( (X[:,i].reshape(-1,1)-self.mean).T , inv(self.covariance) )                                  
            self.delta[i] = np.matmul(term , (X[:,i].reshape(-1,1) - self.mean))
    
    def display(self, pca_components, pca_mean):
        print("Visualizing Mean")
        mean_img = np.dot(self.mean[:,0], pca_components) + pca_mean
        mean_img = np.array(mean_img).astype('uint8')
        mean_img = np.reshape(mean_img,(60,60))
        plt.imshow(mean_img,cmap="gray")
        plt.show()
        print("Visualizing Covariance")
        plt.imshow(self.covariance,cmap="gray")
        plt.show()
        # print("v = ", self.v)                

def t_cost(v, e_h, e_logh):
   I = len(e_h)
   t1 = (v/2) * np.log((v/2))
   t2 = gammaln((v/2))
   finalCost = 0
   for i in range(I):
       t3 = ((v/2) - 1) * e_logh[i]
       t4 = (v/2) * e_h[i]
       finalCost = finalCost + t1 - t2 + t3 - t4
   finalCost = -finalCost
   return finalCost

def apply_pca_and_standardize(data):
    pca = PCA(n_components=100)
    pca.fit(data)
    data_pca = pca.transform(data)
    scaler = StandardScaler()
    scaler.fit(data_pca)
    data_std = scaler.transform(data_pca)
    return data_std,pca

#loading data from pickle dump        
train_f = pickle.load(open("train_f.p", "rb" ))
train_nf = pickle.load(open("train_nf.p", "rb"))
test_f = pickle.load(open("test_f.p", "rb"))
test_nf = pickle.load(open("test_nf.p", "rb"))

#reshaping image from 60,60 to a single vector of 3600 
train_f = train_f.reshape(train_f.shape[0],-1)
train_nf = train_nf.reshape(train_nf.shape[0],-1)
test_f = test_f.reshape(test_f.shape[0],-1)
test_nf = test_nf.reshape(test_nf.shape[0],-1)

#reducing features from 10880 to 100 by principal component analysis
train_f,pca_f =  apply_pca_and_standardize(train_f)
train_nf,pca_nf = apply_pca_and_standardize(train_nf)
test_f,temp = apply_pca_and_standardize(test_f)
test_nf,temp = apply_pca_and_standardize(test_nf)

train_f,train_nf,test_f,test_nf = train_f.T,train_nf.T,test_f.T,test_nf.T

#initializing mean and covariance from dataset
mean_f = np.mean(train_f,axis=1)
covariance_f = np.cov(train_f) * np.eye(train_f.shape[0])
mean_nf = np.mean(train_nf,axis=1)
covariance_nf = np.cov(train_nf) * np.eye(train_nf.shape[0])

#initializing t_distribution model
tdist_f = students_tDist(mean_f.reshape(-1,1), covariance_f, v=10)
tdist_nf = students_tDist(mean_nf.reshape(-1,1), covariance_nf, v=10)

train_size = 100

#Learning the model
for i in range(train_size):
    print("\nPerforming Iteration - {}".format(i))
    print("tdist_for_face")
    tdist_f.apply_EM(train_f)
    print("\ntdist_for_nonface")
    tdist_nf.apply_EM(train_nf)

tdist_f.display(pca_f.components_,pca_f.mean_)
tdist_nf.display(pca_nf.components_,pca_nf.mean_)

P_f_f = np.array([])
P_nf_f = np.array([])
P_f_nf = np.array([])
P_nf_nf = np.array([])

#Running predictions on test data
for i in range(test_size):
    P_f_f = np.append(P_f_f, tdist_f.prob(i, test_f))
    P_f_nf = np.append(P_f_nf, tdist_f.prob(i, test_nf))
    P_nf_f = np.append(P_nf_f, tdist_nf.prob(i,test_f))
    P_nf_nf = np.append(P_nf_nf, tdist_nf.prob(i,test_nf))

#calculating the posterior probabilities        
post_P_f_f = P_f_f / (P_f_f + P_nf_f)
post_P_nf_f = P_nf_f / (P_f_f + P_nf_f)
post_P_f_nf = P_f_nf / (P_f_nf + P_nf_nf)
post_P_nf_nf = P_nf_nf / (P_f_nf + P_nf_nf)


#ROC Curve
plot_ROC_curve(post_P_f_nf, post_P_f_f,test_size)