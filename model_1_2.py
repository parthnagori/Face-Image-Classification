from numpy.linalg import inv, det
import numpy as np
import pickle
from cv1_image_processing import plot_ROC_curve
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
dim = 100
test_size = 100
train_size = 1000
K=1
gaussian_type = sys.argv[1]
if gaussian_type == "mixture":
    K = 3
np.random.seed(2)

class GMM():
    
    def __init__(self,means,covariances,train_size):
        self.train_size = train_size
        self.means = means
        self.covariances = covariances
        self.length = self.means.shape[1]        
        self.weights = np.random.dirichlet(np.ones(K), size = 1)[0]
        self.posterior = np.random.dirichlet(np.ones(K), size = self.train_size)
    
    def pdf(self,k,i,X):
        val1 = np.matmul((X[:,i].reshape(-1,1) - self.means[k]).T, inv(self.covariances[k]))
        val2 = -0.5 * np.matmul(val1,X[:,i].reshape(-1,1) - self.means[k])
        pdf_norm = np.exp(val2) / (np.sqrt(det(self.covariances[k]) * (2*np.pi ** X.shape[0])))
        return pdf_norm
    
    def get_prob(self,i,X):
        val = 0
        for k in range(K):
            val = val + self.weights[k] * self.pdf(k,i,X)
        return val                       
    
    def apply_EM(self,i,k,X):
        #expectation_step
        temp = 0
        for j in range(K):
            temp = temp + self.weights[j] * self.pdf(j,i,X)
        self.posterior[i,k] = self.weights[k] * self.pdf(k,i,X) / temp 
        
        #maximization_step
        #updating weights
        num = 0
        denom = 0
        for i in range(self.train_size):
            num = num + self.posterior[i,k]
            for j in range(0,K):
                denom = denom + self.posterior[i,j]      
        self.weights[k] = 1.0*num/denom 
        
        #updating means
        num = np.zeros((self.length,1))
        denom = 0
        for i in range(self.train_size):
            num = num + self.posterior[i,k] * X[:,i].reshape(-1,1)
            denom = denom + self.posterior[i,k]    
        self.means[k] = 1.0 * num / denom
        
        #updating_covariances
        num = np.zeros((self.length,self.length))
        denom = 0
        for i in range(self.train_size):
            num = num + self.posterior[i,k] * np.matmul( (X[:,i].reshape(-1,1) - self.means[k]) , (X[:,i].reshape(-1,1) - self.means[k]).T )
            denom = denom + self.posterior[i,k]
        self.covariances[k] = 1.0*num/denom    
        self.covariances[k] = np.diag( np.diag(self.covariances[k]) )       
    
    def fit(self,k,X):
        for i in range(self.train_size):
            self.apply_EM(i,k,X)
    
    def display(self, k, pca_components, pca_mean):
        print("Visualizing Mean")
        mean_img = np.dot(self.means[0][:,0], pca_components) + pca_mean
        mean_img = np.array(mean_img).astype('uint8')
        mean_img = np.reshape(mean_img,(60,60))
        plt.imshow(mean_img,cmap="gray")
        plt.show()
        print("Visualizing Covariance")
        plt.imshow(self.covariances[k])
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

#initializing the gmm model for face data
means_f = np.zeros((K,100,1))
covariances_f = np.array([np.random.uniform(low=0.0, high=1.0, size=(dim,dim)) * np.identity(100) for k in range(K)])
gmm_f = GMM(means_f ,covariances_f, train_size)

#initializing the gmm model for nonface data
means_nf = np.zeros((K,100,1))
covariances_nf = np.array([np.random.uniform(low=0.0, high=1.0, size=(dim,dim)) * np.identity(100) for k in range(K)])
gmm_nf = GMM(means_nf ,covariances_nf, train_size)

#learning the model for face data
for k in range(K):
    print("FACE component - {}".format(k))
    gmm_f.fit(k,train_f)
    print("gmm_f -> {}".format(gmm_f.weights))

#learning the model for non face data
for k in range(K):    
    print("NON-FACE component - {}".format(k))
    gmm_nf.fit(k,train_nf)    
    print("gmm_nf -> {}".format(gmm_nf.weights))

#displaying the learned mean and covariance for both face and non face data
gmm_f.display(0,pca_f.components_,pca_f.mean_)
gmm_nf.display(0,pca_nf.components_,pca_nf.mean_)

P_f_f = np.array([])
P_nf_f = np.array([])
P_f_nf = np.array([])
P_nf_nf = np.array([])

for i in range(test_size):
    P_f_f = np.append( P_f_f , gmm_f.get_prob(i,test_f) )
    P_nf_f = np.append( P_nf_f , gmm_nf.get_prob(i,test_f) )
    P_f_nf = np.append( P_f_nf , gmm_f.get_prob(i,test_nf) )
    P_nf_nf = np.append( P_nf_nf , gmm_nf.get_prob(i,test_nf) )

post_P_f_f = P_f_f/( P_f_f + P_nf_f )
post_P_nf_f = P_nf_f/( P_f_f + P_nf_f )
post_P_f_nf = P_f_nf/( P_f_nf + P_nf_nf )
post_P_nf_nf = P_nf_nf/( P_f_nf + P_nf_nf )

#ROC Curve
plot_ROC_curve(post_P_f_nf, post_P_f_f,test_size)


