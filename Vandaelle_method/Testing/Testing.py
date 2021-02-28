import sys
sys.path.append('../')
from Predict.Predict import Predict
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



def Test(TrainedModel, Image_Folder, landmark_data,train_num ,i):
    # first calculate the average center as this will only need to be calc once
    x1landmark = landmark_data.iloc[:,1]
    y1landmark = landmark_data.iloc[:,2]
    x1landmark_mean, y1landmark_mean = np.mean(x1landmark), np.mean(y1landmark)
    center = (x1landmark_mean, y1landmark_mean)
    # create an empty list (distances) to append the distances errors to; abs(landmark - prediction)
    distances = []
    counter = 0
    landmark_data = landmark_data.iloc[:6,:]
    #print('##########')
    #print(landmark_data)
    dir = os.listdir(Image_Folder)
    #print(dir)
    for i, filename in enumerate(dir):#range(len(landmark_data)):
        PIL_image = Image.open(Image_Folder + filename)
        

        #xerror = random.choice((-1,1))*random.randint(2,50)
        #yerror = random.choice((-1,1))*random.randint(2,50)
        #error = random.randint(0,9)
        #center = (landmark_data.iloc[i,1] + xerror, landmark_data.iloc[i,2] + yerror)
        true_landmark = (float(landmark_data.iloc[i,1]),float(landmark_data.iloc[i,2]))
        #print(true_landmark)
        prediction = Predict(TrainedModel, PIL_image, center, true_landmark = true_landmark, feature_types=['type-4' , 'type-2-x', 'type-2-y','type-3-x', 'type-3-y'], radius = 200, Nsamples = 1000, sampling_distribution = 'uniform', crop_size = 50, resolution = 5, finalpredstat = 'median', pos_samples = 40)
        #print('hello')
        prediction.sample()
        #print(prediction.xwithin)
        #print('hello')
        
        #print(len(prediction.xwithin))

        #print('here')
        #plt.figure(figsize=(10,10))
        prediction.forward()
        #print('done')
        plt.figure(figsize = (10,10))
        plt.imshow(PIL_image)
        plt.scatter(prediction.xwithin,prediction.ywithin, 0.5, color='r')
        #plt.scatter(prediction.posLandmarks_Xcoords, prediction.posLandmarks_Ycoords, color = 'b')
        #plt.scatter(true_landmark[0],true_landmark[1])
        plt.savefig('C:/Users/dylan/Work-Projects/msc_haar/image_predictions/{}sample1_2tier.png'.format(landmark_data.iloc[i,0][:-4] + '{}'.format(i)))
        plt.show()
        #plt.imshow(PIL_image)
        #plt.scatter(prediction.pred_coord[0],prediction.pred_coord[1])
        #plt.show()
        counter += 1
        #print(counter)
        

        if prediction.error == 0:
            print( landmark_data.iloc[i,0])
        else:
            pred2 = Predict(TrainedModel, PIL_image, prediction.pred_coord, true_landmark=true_landmark,feature_types=['type-4' , 'type-2-x', 'type-2-y','type-3-x', 'type-3-y'] , radius = 20, Nsamples = 1000, sampling_distribution = 'normal', crop_size = 50, resolution = 5, finalpredstat = 'median', pos_samples = 40)
            pred2.sample()
            pred2.forward()
            #plt.figure(figsize = (10,10))
            #plt.imshow(PIL_image)
            #plt.scatter(pred2.xwithin,pred2.ywithin, 0.5, color='r')
            #plt.scatter(pred2.posLandmarks_Xcoords, pred2.posLandmarks_Ycoords, color='b')
            #plt.scatter(pred2.pred_coord[0],pred2.pred_coord[1], color = 'b')
            #plt.scatter(true_landmark[0],true_landmark[1])
            #plt.savefig('C:/Users/dylan/Work-Projects/msc_haar/image_predictions/{}sample2_2tier.png'.format(landmark_data.iloc[i,0][:-4]  + '{}'.format(i)))
            #plt.show()
            '''
            if pred2.error == 0:
                print(landmark_data.iloc[i,0])
            else:
                pred3 = Predict(TrainedModel, PIL_image, pred2.pred_coord,true_landmark = true_landmark, feature_types=['type-4' , 'type-2-x', 'type-2-y','type-3-x', 'type-3-y'],radius = 20, Nsamples = 1000, sampling_distribution = 'normal', crop_size = 50, resolution = 5, finalpredstat = 'median', pos_samples = 50)
                pred3.sample()
                pred3.forward()

        #plt.figure(figsize = (10,10))
        ##plt.imshow(PIL_image)
        #plt.scatter(pred3.xwithin,pred3.ywithin, 0.5, color = 'r')
        #plt.scatter(pred3.posLandmarks_Xcoords, pred3.posLandmarks_Ycoords, color='b')
        #plt.savefig('sample3.png')
        #plt.show()
                plt.figure(figsize = (10,10))
                plt.imshow(PIL_image)
                plt.scatter(pred2.pred_coord[0],pred2.pred_coord[1], color = 'b')
                plt.scatter(true_landmark[0],true_landmark[1])
                plt.savefig('C:/Users/dylan/Work-Projects/msc_haar/image_predictions/{}sample3_2tier.png'.format(landmark_data.iloc[i,0][:-4]))
                #plt.show()
            '''
        #count_pos,x_pred,y_pred, posLandmarks_Xcoords,posLandmarks_Ycoords  = Predict_For_TestOutput(TrainedModel, PIL_image, center)
        
            distances.append([pred2.pos_sample_count ,landmark_data.iloc[i,0],true_landmark[0],true_landmark[1],pred2.pred_coord[0],pred2.pred_coord[1], pred2.error])
    distances = pd.DataFrame(distances, columns = ['No._pos_samples_averaged','Image_name','truex','truey','predx','predy','distance' ])
    return(distances)











     
     
