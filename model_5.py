import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from cv1_image_processing import plot_ROC_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_size = 1000
test_size = 100
K = 40
D = 100
# np.random.seed(20)

class FactorAnalyzer():

    def __init__(self,mu,covariance,phi):
        self.mean = mu
        self.covariance = covariance
        self.phi = phi
        self.E_h = np.zeros((train_size,K,D))
        self.E_hi_hT = np.zeros((train_size,K,K))
    
    def prob(self, i_index, X):
        sigma = np.matmul(self.phi,self.phi.T) + self.covariance
        term1 = -0.5 * (X[:,i_index].reshape(-1,1) - self.mean).T
        term2 = inv(sigma)
        term3 = X[:,i_index].reshape(-1,1) - self.mean
        expo1 = np.matmul(term1,term2)
        expo2 = np.matmul(expo1,term3)[0,0]
        val =  np.exp(expo2)
        det_sigma = det(sigma)
        if det_sigma < 0:
            det_sigma = -det_sigma
        val =  val / np.sqrt(det_sigma)
        return val
    
    def apply_EM(self,X):
        #expecting
        t1 = np.matmul(self.phi.T, inv(self.covariance))
        t2 = np.matmul(t1, self.phi) + np.eye(K)
        t3 = np.matmul(inv(t2), self.phi.T)
        t4 = np.matmul(t3, inv(self.covariance))
        for i in range(0,train_size):
            self.E_h[i] = np.matmul(t4, X[:,i] - self.mean)
            self.E_hi_hT[i] = inv(t2) + np.matmul(self.E_h[i],self.E_h[i].T)

        #maximizing
        #updating phi value
        t1 = np.zeros((100,K))
        t2 = np.zeros((K,K))
        for i in range(0,train_size):
            t1 = t1 + np.matmul( (X[:,i]-self.mean) , self.E_h[i].T ) 
            t2 = t2 + self.E_hi_hT[i]
        self.phi = np.matmul(t1,t2)    
        
        #updating the covariance
        temp = np.zeros((D,D))
        for i in range(0,train_size):
            temp = temp + np.matmul(X[:,i].reshape(-1,1)-self.mean, (X[:,i].reshape(-1,1)-self.mean).T)
            t2 = np.matmul(self.phi,self.E_h[i])
            t3 = np.matmul(t2, (X[:,i].reshape(-1,1)-self.mean ))
            temp = temp - t3
        temp = temp/train_size
        self.covariance = np.diag( np.diag(temp) )
        
    def display(self, pca_components, pca_mean):
        print("Visualizing Mean")
        mean_img = np.dot(self.mean[:,0], pca_components) + pca_mean
        mean_img = np.array(mean_img).astype('uint8')
        mean_img = np.reshape(mean_img,(60,60))
        plt.imshow(mean_img,cmap="gray")
        plt.show()
        print("Visualizing Covariance")
        plt.imshow(self.covariance)
        plt.show()
    
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

#initializing the mean, phi and covariance for face data
mean_f = np.mean(train_f,axis=1)
phi_f = np.random.rand(D,K)
covariance_f = np.random.rand(D, D)
covariance_f = np.diag(np.diag(covariance_f))

#initializing the mean, phi and covariance for non face data
mean_nf = np.mean(train_nf,axis=1)
phi_nf = np.random.rand(D,K)
covariance_nf = np.random.rand(D, D)
covariance_nf = np.diag(np.diag(covariance_nf))


fAnalyzer_f = FactorAnalyzer(mean_f.reshape(-1,1), covariance_f, phi_f)
fAnalyzer_nf = FactorAnalyzer(mean_nf.reshape(-1,1), covariance_nf, phi_nf)


n_iter=100

for i in range(0,n_iter):
    print("Performing iteration no. {}".format(i))
    fAnalyzer_f.apply_EM(train_f)
    fAnalyzer_nf.apply_EM(train_nf)
      
fAnalyzer_f.display(pca_f.components_,pca_f.mean_)
fAnalyzer_nf.display(pca_nf.components_,pca_nf.mean_)

P_f_f = np.array([])
P_nf_f = np.array([])
P_f_nf = np.array([])
P_nf_nf = np.array([])

#Running predictions on test data
for i in range(test_size):
    P_f_f = np.append(P_f_f, fAnalyzer_f.prob(i, test_f))
    P_f_nf = np.append(P_f_nf, fAnalyzer_f.prob(i, test_nf))
    P_nf_f = np.append(P_nf_f, fAnalyzer_nf.prob(i,test_f))
    P_nf_nf = np.append(P_nf_nf, fAnalyzer_nf.prob(i,test_nf))
        
post_P_f_f = P_f_f / (P_f_f + P_nf_f)
post_P_nf_f = P_nf_f / (P_f_f + P_nf_f)
post_P_f_nf = P_f_nf / (P_f_nf + P_nf_nf)
post_P_nf_nf = P_nf_nf / (P_f_nf + P_nf_nf)

#ROC Curve
plot_ROC_curve(post_P_f_nf, post_P_f_f,test_size)

