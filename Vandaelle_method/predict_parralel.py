import sys
sys.path.append('C:/Users/dylan/Work-Projects/msc_haar/final-project')
#print(sys.path)
import os
#from Testing.Testing import Test
from Predict.Predict import Predict
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def find_landmarks(TrainedModel, Image_Folder, landmark_data, feature_types, feature_coords, crop_size, resolution ,count):
    x1landmark = landmark_data.iloc[:350,1]
    y1landmark = landmark_data.iloc[:350,2]
    x1landmark_mean, y1landmark_mean = np.mean(x1landmark), np.mean(y1landmark)
    center = (x1landmark_mean, y1landmark_mean)
    images = os.listdir(Image_Folder)
    #print(images)
    data = []
    #for image in images[:]:
    #    PIL_image = Image.open(Image_Folder + image)
    for i in range(landmark_data.iloc[800:1000,:].shape[0]):
        print('.')
        image = landmark_data.iloc[i,0]
        PIL_image = Image.open(Image_Folder  + landmark_data.iloc[i,0])

        true_landmark = (landmark_data.iloc[i,1], landmark_data.iloc[i,2])

        prediction = Predict(TrainedModel.modelobject, PIL_image, center,true_landmark= true_landmark, feature_types = feature_types, feature_coords = feature_coords, radius = 250, Nsamples = 1000, sampling_distribution = 'normal', crop_size = crop_size, resolution=resolution, finalpredstat = 'median', pos_samples = 60)
        prediction.sample()
        prediction.forward()

        if prediction.no_pred == 1:
            print(image)
        else:
            pred2 = Predict(TrainedModel.modelobject, PIL_image, prediction.pred_coord,true_landmark= true_landmark,feature_types = feature_types, feature_coords=feature_coords , radius = 40, Nsamples = 1000, sampling_distribution = 'normal',crop_size=crop_size , resolution=resolution, finalpredstat = 'median', pos_samples = 40)
            pred2.sample()
            pred2.forward()
            if pred2.no_pred == 1:
                print(image)
            else:
                data.append([pred2.pos_sample_count ,image,pred2.pred_coord[0],pred2.pred_coord[1], pred2.error])
    data = pd.DataFrame(data, columns = ['No._pos_samples_averaged','Image_name','predx','predy', 'error' ])
    data.to_csv('predictions{}.csv'.format(count))
    print('done')

if __name__ == "__main__":
    def callback(results):
        print(results)
    def error_callback(error):
        print(str(error))
    pool = mp.Pool(mp.cpu_count())
    files = os.listdir('C:/Users/dylan/Work-Projects/msc_haar/final-project/models')

    dir = 'C:/Users/dylan/Work-Projects/msc_haar/final-project/models/'

    y = np.array([[1,3], [3, 5], [5, 7], [7, 9], [9, 11], [11,13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]])
    bad_wings_dir = 'C:/Users/dylan/Work-Projects/msc_haar/final-project/bad_wings/'
    C_data = 'C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/images_left/'
    landmark_data = pd.read_csv('C:/Users/dylan/Work-Projects/msc_haar/tsetsedata_2019_left_commas/annotations_left.txt')
    print(landmark_data.iloc[1,0])
    for count,i in enumerate(y,0):
        object = open(dir +'train_class{}.pkl'.format(count) , 'rb')
        model = pickle.load(object)
        crop_size = model.crop_size
        resolution = model.resolution
        feature_coords = model.feature_coords
        feature_types = model.feature_types
        print('yes')
        pool.apply_async(find_landmarks, args = [model,C_data, pd.concat([landmark_data.iloc[:,0],landmark_data.iloc[:,i[0]:i[1]]], axis=1), feature_types,feature_coords, crop_size, resolution, count] , error_callback = error_callback )
    
    pool.close() 
    pool.join()


    



