
import sys
sys.path.append('..')
from Training.Training_ import Train
import pandas as pd
import numpy as np
import pickle
import os


# directories
C_data =  'C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/images_left/'
ldata = pd.read_csv('C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/annotations_left.txt'  )
#landmark_data = pd.concat([ldata.iloc[:20,0], ldata.iloc[:20,1:3]], axis = 1)
'''
print(landmark_data)

train = Train(C_data,landmark_data, resolution = 8)
train.sample()
train.extract_haar_features()
train.fit()
train.feature_select()
train.extract_haar_features(feature_count=train.sig_feature_count)
print('training again')
train.fit()
'''
#y = [[0,2,[8]],[2,4,[8]],[4,6,[8]],[6,8,[8]],[8,10,[8]],[10,12,[8]],[12,14,[8]],[14,16,[8]],[16,18,[8]],[18,20,[8]], [20,22,[8]]]
#res_params = [8,8,8,8,8,8,8,8,8,8,8,8]
def train_all(params, ldata, C_data):
    for count,i in enumerate(params):
        landmark_data = landmark_data = pd.concat([ldata.iloc[:,0], ldata.iloc[:,i[0]:i[1]]], axis = 1)
        ld = landmark_data.iloc[:800,:].sample(60)
        pre_train = Train(C_data ,ld, resolution = i[2][0])
        pre_train.sample()
        pre_train.extract_haar_features()
        pre_train.fit()

        
        pre_train.feature_select()
        train = Train(C_data,landmark_data.iloc[:800,:], resolution = i[2][0])
        train.sample()
        train.feature_types = pre_train.feature_types
        train.feature_coords = pre_train.feature_coords
        train.extract_haar_features(feature_count = pre_train.sig_feature_count )
        train.fit()
        filename = 'C:/Users/dylan/Work-Projects/msc_haar/final-project/models/train_class{}.pkl'.format(count)
        pickle.dump(train, open(filename, 'wb'))
        print('done')
     

if __name__ == "__main__":
    #y = [[0,2,[8]],[2,4,[8]],[4,6,[8]],[6,8,[8]],[8,10,[8]],[10,12,[8]],[12,14,[8]],[14,16,[8]],[16,18,[8]],[18,20,[8]], [20,22,[8]]]
    y = [[1,3,[7]],[3,5,[7]],[5,7,[8]],[7,9,[8]],[9,11,[7]],[11,13,[8]],[13,15,[7]],[15,17,[7]],[17,19,[7]],[19,21,[7]], [21,23,[7]]]
    
    C_data = 'C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/images_left/'
    ld_data = pd.DataFrame(pd.read_csv('C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/annotations_left.txt'))
    #landmark_data = pd.concat([pd.DataFrame(os.listdir(C_data)), ld_data], axis = 1)
    #print(ld_data)
    train_all(y, ld_data, C_data)


    




