import numpy as np
from numpy.linalg import inv,det
from scipy.special import gamma,digamma
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import optimize
import cv2
import pickle
from cv1_image_processing import plot_ROC_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


train_size = 1000
test_size= 100
D = 100
K = 3

E_h = np.zeros((K,train_size))
E_log_h = np.zeros((K,train_size))
delta = np.zeros((K,train_size))      


def get_delta(X,i,k,mu,covariance):
    term1 = np.matmul( (X[:,i].reshape(-1,1)-mu[k]).T,inv(covariance[k]) )                                  
    term2 = np.matmul(term1,(X[:,i].reshape(-1,1) - mu[k]))
    return term2    
    
def prob(i,k,v,mu,covariance,X):
    D = mu[k].shape[0]
    c1 = gamma( (v[k] + D)/2.0 ) / ( pow( (v[k] * np.pi), D/2 )*np.sqrt(det(covariance[k]))*gamma(v[k]/2))
    term2 = get_delta(X,i,k,mu,covariance)    
    c2 = (1 + term2/v[k])
    val = c1 * pow(c2, -(v[k]+D)/2)
    return val[0,0] #* 100000.0

def get_prob(i,v,mu,covariance,X):
    val = 0
    for k in range(0,K):
        val = val + prob(i,k,v,mu,covariance,X)
    return val
    
def get_E_hi(i,k,v,mu,covariance,X):
    D = mu.shape[1]
    term1 = np.matmul((X[:,i].reshape(-1,1)-mu[k]).T , inv(covariance[k]))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1)-mu[k])[0,0]
    val = (v[k] + D) / (v[k] + term2)
    return val
    
def get_E_log_hi(i,k,v,mu,covariance,X):
    D = mu.shape[1]
    term1 = np.matmul((X[:,i].reshape(-1,1)-mu[k]).T , inv(covariance[k]))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1)-mu[k])[0,0]
    val = digamma((v[k]+D)/2) - np.log( (v[k] + term2)/2 )
    return val

def tCost0(v):
    global E_h, E_log_h 
    val = 0
    for i in range(0,train_size):
       val = val + ( (v[0]/2) - 1)*E_log_h[0,i] - (v[0]/2)*E_h[0,i] - (v[0]/2)*np.log(v[0]/2) - np.log(gamma(v[0]/2))                 
    return -val

def tCost1(v):
    global E_h, E_log_h 
    val = 0
    for i in range(0,train_size):
       val = val + ( (v[1]/2) - 1)*E_log_h[1,i] - (v[1]/2)*E_h[1,i] - (v[1]/2)*np.log(v[1]/2) - np.log(gamma(v[1]/2))                 
    return -val

def tCost2(v):
    global E_h, E_log_h 
    val = 0
    for i in range(0,train_size):
       val = val + ( (v[2]/2) - 1)*E_log_h[2,i] - (v[2]/2)*E_h[2,i] - (v[2]/2)*np.log(v[2]/2) - np.log(gamma(v[2]/2))                 
    return -val


def E_step(v,k,mu,covariance,X):
    global E_h,E_log_h,delta
    for i in range(0,train_size):
        term = np.matmul( (X[:,i].reshape(-1,1)-mu[k]).T , inv(covariance[k]) )                                  
        delta[k,i] = np.matmul(term , (X[:,i].reshape(-1,1) - mu[k]))
        E_h[k,i] = get_E_hi(i,k,v,mu,covariance,X)
        E_log_h[k,i] = get_E_log_hi(i,k,v,mu,covariance,X)
    return [delta, E_h, E_log_h]
        

def run_one_EM_step(v,k,mu,covariance,X):
    D = mu.shape[1]
    #global delta, E_h, E_log_h    

    #expecting
    [delta, E_h, E_log_h] = E_step(v,k,mu,covariance,X)
    
    #Updating Mean            
    temp_mean = np.zeros((D,1))
    denom = 0
    for i in range(0,train_size):
        temp_mean = temp_mean + E_h[k,i]*X[:,i].reshape(-1,1)
        denom = denom + E_h[k,i]
    mu[k] = temp_mean/denom    
        
    #Updating Variance
    num = np.zeros((D,D))
    for i in range(0,train_size):
        prod = np.matmul( (X[:,i].reshape(-1,1) - mu[k]) , (X[:,i].reshape(-1,1) - mu[k]).T )
        num = num + E_h[k,i]*prod
    covariance[k] = num/denom
    covariance[k] = np.diag( np.diag(covariance[k]) )
      

    #calculating argmin v
    if k == 0:
       v[k] = optimize.fmin(tCost0,[50,50,50])[0]             
    if k == 1:
       v[k] = optimize.fmin(tCost1,[50,50,50])[0]         
    if k == 2:
       v[k] = optimize.fmin(tCost2,[50,50,50])[0]         
    
    return [v, mu, covariance]  

def apply_EM(v,mu,covariance,X):
    for k in range(0,K):
        [v, mu, covariance] = run_one_EM_step(v,k,mu,covariance,X)
    return [v,mu,covariance]    


def display(mean_data, covariance_data, pca_components, pca_mean):
    print("Visualizing Mean")
    mean_img = np.dot(mean_data[:,0], pca_components) + pca_mean
    mean_img = np.array(mean_img).astype('uint8')
    mean_img = np.reshape(mean_img,(60,60))
    plt.imshow(mean_img,cmap="gray")
    plt.show()
    print("Visualizing Covariance")
    plt.imshow(covariance_data)
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


v_face = [50,50,50]
means_f = np.random.rand(K,D,1)
means_nf = np.random.rand(K,D,1)

v_nonface = [50,50,50]
covariances_f = np.random.rand(K,D,D)
covariances_nf = np.random.rand(K,D,D)

train_size = 1000
n_iter = 5
k = 0

#Learning mixture of t distributions for face data
for i in range(0,n_iter):
    print("Performing iteration {} for face".format(i))
    [v_face,means_f,covariances_f] = apply_EM(v_face,means_f, covariances_f, train_f)

display(means_f[0], covariances_f[0], pca_f.components_,pca_f.mean_)


E_h = np.zeros((K,train_size))
E_log_h = np.zeros((K,train_size))
delta = np.zeros((K,train_size))  

#Learning mixture of t distributions for face data
for i in range(0,n_iter):
    print("Performing iteration {} for non face".format(i))
    [v_nonface,means_nf,covariances_nf] = apply_EM(v_nonface,means_nf, covariances_nf, train_nf)
 
display(means_nf[0], covariances_nf[0], pca_nf.components_,pca_nf.mean_)

#Running predictions on test data
P_f_f = np.array([])
P_nf_f = np.array([])
P_f_nf = np.array([])
P_nf_nf = np.array([])


for i in range(0,test_size):
    P_f_f = np.append( P_f_f , get_prob(i,v_face,means_f,covariances_f,test_f) )
    P_f_nf = np.append( P_f_nf , get_prob(i,v_face,means_f,covariances_f,test_nf) )    
    P_nf_f = np.append( P_nf_f , get_prob(i,v_nonface,means_nf,covariances_nf,test_f) )
    P_nf_nf = np.append( P_nf_nf , get_prob(i,v_nonface,means_nf,covariances_nf,test_nf) )

post_P_f_f = P_f_f/( P_f_f + P_nf_f )
post_P_nf_f = P_nf_f/( P_f_f + P_nf_f )
post_P_f_nf = P_f_nf/( P_f_nf + P_nf_nf )
post_P_nf_nf = P_nf_nf/( P_f_nf + P_nf_nf )

#ROC Curve
plot_ROC_curve(post_P_f_nf, post_P_f_f,100)
