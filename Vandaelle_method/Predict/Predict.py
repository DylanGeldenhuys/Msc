import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import stats
#import utilities
import sys
from utilities.utilities import  rgb2gray, get_window, FeatureExtract
from numpy import asarray
import math
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

        



class Predict():

    def __init__(self,trained_ModelObject, PIL_image, center, feature_types,true_landmark = None, feature_coords = None, radius = 50, Nsamples = 1000, sampling_distribution = 'uniform', crop_size = 50, resolution = 5, finalpredstat = 'median', pos_samples = 50):
        self.trained_ModelObject = trained_ModelObject
        self.true_landmark = true_landmark
        self.PIL_image = PIL_image
        self.pos_samples = pos_samples
        self.center = center
        self.radius = radius
        self.Nsamples = Nsamples
        self.sampling_distribution = sampling_distribution
        self.crop_size = crop_size
        self.resolution = resolution
        self.data  = []
        self.feature_types = feature_types
        self.feature_coords = feature_coords
        self.pos_sample_count = 0
        self.finalpredstat = finalpredstat
        

    def sample(self):
        if self.sampling_distribution == 'uniform':
            radians = np.random.uniform(0,1,self.Nsamples) * 2 * np.pi
            r = np.random.uniform(0,self.radius,self.Nsamples)

        elif self.sampling_distribution == 'normal':
            radians = np.random.normal(0,1,self.Nsamples) * 2 * np.pi
            r = np.random.normal(0,self.radius,self.Nsamples) 
        #print(r * np.cos(radians) + [self.center[0]]*len(radians))

        self.xwithin = [sum(i) for i in zip(r * np.cos(radians), [self.center[0]]*len(radians))]
    
        self.ywithin =  [sum(i) for i in zip(r * np.sin(radians), [self.center[1]]*len(radians))]
        #self.preds[:,0] = np.array([self.xwithin, self.ywithin]).T
    
    def forward(self):
        self.xsampled = []
        self.ysampled = []
        if self.sampling_distribution == 'uniform':
            radians = np.random.uniform(0,1,self.Nsamples) * 2 * np.pi
            r = np.random.uniform(0,self.radius,self.Nsamples)

        elif self.sampling_distribution == 'normal':
            radians = np.random.normal(0,1,self.Nsamples) * 2 * np.pi
            r = np.random.normal(0,self.radius,self.Nsamples) 

        for x,y in zip(np.array(r * np.cos(radians)) +  np.array([self.center[0]]*len(radians)),np.array(r * np.sin(radians)) +  np.array([self.center[1]]*len(radians))):

            if self.pos_sample_count < self.pos_samples:
                self.xsampled.append(x)
                self.ysampled.append(y)
                features = FeatureExtract(rgb2gray(asarray(get_window(self.PIL_image, (x,y), self.crop_size, self.resolution))), self.feature_types, self.feature_coords)
                features = pd.DataFrame([features])
                class_prediction = self.trained_ModelObject.predict(features)
                
                if class_prediction == 'pos':
                #print(class_prediction)
                    self.data.append([x,y,features,1])
                    self.pos_sample_count += 1
                else:
                    self.data.append([x,y,features,0])
            else:
                break
        if self.pos_sample_count != 0:
            self.data = np.array(self.data, dtype=object)

            posLandmarks = self.data[self.data[:,3] == 1] 
            self.posLandmarks_Xcoords = posLandmarks[:,0]
            self.posLandmarks_Ycoords = posLandmarks[:,1]
            if self.finalpredstat == 'median':
                self.pred_coord = (stats.median(self.posLandmarks_Xcoords),stats.median(self.posLandmarks_Ycoords))
            if self.finalpredstat == 'mean':
                self.pred_coord = (stats.mean(self.posLandmarks_Xcoords),stats.mean(self.posLandmarks_Ycoords))
            if self.true_landmark != None:
                self.error = float(math.sqrt((abs(self.true_landmark[0]- self.pred_coord[0]))**2 + (abs(self.true_landmark[1]- self.pred_coord[1]))**2 ))
                self.no_pred = 0
        else:
            self.no_pred = 1

    '''
    def forward(self):
        for i, (x,y) in enumerate(zip(self.xwithin,self.ywithin)):
            if self.pos_sample_count < self.pos_samples:
            #print(self.pos_sample_count)
            
            #plt.imshow(get_window(self.PIL_image, (x,y), self.crop_size, 50))
            #plt.show()
                
                features = FeatureExtract(rgb2gray(asarray(get_window(self.PIL_image, (x,y), self.crop_size, self.resolution))), self.feature_types, self.feature_coords)
                features = pd.DataFrame([features])
                class_prediction = self.trained_ModelObject.predict(features)
                
                if class_prediction == 'pos':
                #print(class_prediction)
                    self.data.append([x,y,features,1])
                    self.pos_sample_count += 1
                else:
                    self.data.append([x,y,features,0])
            else:
                break
        if self.pos_sample_count != 0:
            self.data = np.array(self.data)
            #print('##############################################################')
            #print(self.data)

            posLandmarks = self.data[self.data[:,3] == 1] 
            self.posLandmarks_Xcoords = posLandmarks[:,0]
            self.posLandmarks_Ycoords = posLandmarks[:,1]
            if self.finalpredstat == 'median':
                self.pred_coord = (stats.median(self.posLandmarks_Xcoords),stats.median(self.posLandmarks_Ycoords))
            if self.finalpredstat == 'mean':
                self.pred_coord = (stats.mean(self.posLandmarks_Xcoords),stats.mean(self.posLandmarks_Ycoords))
            self.error = float(math.sqrt((abs(self.true_landmark[0]- self.pred_coord[0]))**2 + (abs(self.true_landmark[1]- self.pred_coord[1]))**2 ))
        else:
            self.error = 0
    
    '''
        


        

        
