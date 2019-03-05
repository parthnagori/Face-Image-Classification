import numpy as np
import pandas as pd
import cv2
import random
import pickle
import glob
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to calculate IoU Ratio

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# Extracting Face & Non Face Images (Crop and Resize 60X60) - 1100 Images
def extract_images(train_f_path, train_nf_path, test_f_path, test_nf_path, root_path, df, umd_path):
    persons = set()
    cnt = 0
    f_path = train_f_path
    nf_path = train_nf_path
    for index, row in df.iterrows():
        x,y,w,h = (int(val) for val in row[['FACE_X', 'FACE_Y','FACE_WIDTH', 'FACE_HEIGHT']])
        img_file = row['FILE']
        img_path = umd_path+img_file
        person = img_file.split("/")[0]
        if person not in persons and cnt < 1100:
            if cnt > 999:
                f_path = test_f_path
                nf_path = test_nf_path
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')
            max_x, max_y = img.shape
            if max_x >= 500 and max_y >= 500:
                f_img = img[y:y+h, x:x+w]
                resized_f_img = cv2.resize(f_img, (60, 60))           
                cv2.imwrite(f_path+"image_f_"+str(cnt)+".jpg", resized_f_img)
                #cropping non face image from the same face image's background
                boxA = [x,y,x+w,y+h]
                iou = 1
                ishape = (0,0,0)
                #checking for IoU < 0.3 and shape to be (60,60,3)
                while iou >= 0 or ishape != (60, 60) :
                    x1, y1 = random.randint(0,max_x-60), random.randint(0,max_y-60)
                    boxB = [x1,y1,x1+60,y1+60]
                    iou = bb_intersection_over_union(boxA, boxB)
                    nf_img = img[y1:y1+60,x1:x1+60]
                    ishape = nf_img.shape
                cv2.imwrite(nf_path+"image_nf_"+str(cnt)+".jpg", nf_img)
                persons.add(person)
                cnt+=1
                

# Load images to Numpy array and create a dump using pickle
def load_and_dump_dataset(root_path, train_f_path, train_nf_path, test_f_path, test_nf_path):
    train_f = np.array([np.array(Image.open(train_f_path+"image_f_"+str(i)+".jpg")) for i in range(1000)])
    train_nf = np.array([np.array(Image.open(train_nf_path+"image_nf_"+str(i)+".jpg")) for i in range(1000)])
    test_f = np.array([np.array(Image.open(test_f_path+"image_f_"+str(i)+".jpg")) for i in range(1000,1100)])
    test_nf = np.array([np.array(Image.open(test_nf_path+"image_nf_"+str(i)+".jpg")) for i in range(1000,1100)])
    pickle.dump( train_f, open(root_path+"/train_f.p", "wb+"))
    pickle.dump( train_nf, open(root_path+"/train_nf.p", "wb+"))
    pickle.dump( test_f, open(root_path+"/test_f.p", "wb+"))
    pickle.dump( test_nf, open(root_path+"/test_nf.p", "wb+"))


def plot_ROC_curve(post_f_nf, post_f_f,size):
    predictions = np.append(post_f_nf, post_f_f)
    temp1 = [0]*size
    temp2 = [1]*size
    actual = np.append(temp1,temp2)
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.show()


def initialize(extract=False):
    # root_path = "/Users/parthnagori/cv_proj_1"
    root_path = "pnagori_project01"
    umd_path = root_path+"/umdfaces_batch3/"
    umd_csv = "/umdfaces_batch3_ultraface.csv"
    train_f_path = root_path+"/train_f/"
    train_nf_path = root_path+"/train_nf/"
    test_f_path = root_path+"/test_f/"
    test_nf_path = root_path+"/test_nf/"
    if extract:
        df = pd.read_csv(umd_path+umd_csv)
        cols = ['FILE', 'FACE_X', 'FACE_Y','FACE_WIDTH', 'FACE_HEIGHT']
        df = df[cols]
        extract_images(train_f_path, train_nf_path, test_f_path, test_nf_path, root_path, df, umd_path)
        load_and_dump_dataset(root_path, train_f_path, train_nf_path, test_f_path, test_nf_path)
    
    

if __name__ == "__main__":
    initialize(False)
    





    

    



