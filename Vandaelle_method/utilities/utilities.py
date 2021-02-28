
from PIL import Image
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
import pandas as pd
import numpy as np
from numpy import asarray
import datetime



def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_window(PIL_image, sample_coord ,  crop_size = 50, resolution = 5):
    x,y = sample_coord
    img_cropped = PIL_image.crop((x- crop_size,y-crop_size,x+crop_size,y+crop_size) )
    #plt.show(img_cropped)
    #plt.show()
    img_cropped.thumbnail([resolution,resolution])
    return img_cropped


feature_types = ['type-4' , 'type-2-x', 'type-2-y','type-3-x', 'type-3-y']


#def FeatureExtract(img, feature_types, feature_coord=None):
#    """Extract the haar feature for the current image"""
#    ii = integral_image(img)
#    features = []
#    for i in range(len(feature_types)):
#        features.extend(haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
#                             feature_types[i],
#                             feature_coord=None))
#    return features


def FeatureExtract(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

def pix_sampling_training(imgname, image, Xcoord_Landmark, Ycoord_Landmark, R=9,Rmax=300, NposPIX=100): #, imgname, i):

    coord_label_df = pd.DataFrame(columns=['label', 'coord', 'Image_name'])

    
    NnegPIX = 2*NposPIX

    a = np.random.uniform(0,1,NposPIX) * 2 * np.pi 
    r = R * np.sqrt(np.random.uniform(0,1,NposPIX)) 
    

    ## If you need it in Cartesian coordinatess
    xwithin = r * np.cos(a)
    ywithin = r * np.sin(a)

    outer_radius =Rmax*Rmax
    inner_radius = R*R
    
    rho= np.sqrt(np.random.uniform(inner_radius, 
                             outer_radius, size=NnegPIX))
    

    theta=  np.random.uniform( 0, 2*np.pi, NnegPIX)
    xhoop = rho * np.cos(theta)
    yhoop = rho * np.sin(theta)

    for i in range(len(xwithin)):
        df2_within = pd.DataFrame({'label': ['pos'], 'coord': [(Xcoord_Landmark + 
                xwithin[i],Ycoord_Landmark + ywithin[i])], 'Image_name':[imgname]})
        coord_label_df = coord_label_df.append(df2_within)
        
    for i in range(len(xhoop)):
        df2_hoop = pd.DataFrame({'label': ['neg'], 'coord': [(Xcoord_Landmark +
                xhoop[i], Ycoord_Landmark +yhoop[i])], 'Image_name': [imgname]})
        coord_label_df = coord_label_df.append(df2_hoop)

    return(coord_label_df)


'''
def generate_sample_df(Image_Folder, DataFrame, R = 9, Rmax = 300, NposPIX =100):
    df = pd.DataFrame(columns=['label', 'coord', 'Image_name'])
    for i in range(1,len(DataFrame)):
        imageName = DataFrame.iloc[i,0]
        image = Image.open(Image_Folder + imageName)
        
        landmark_Xcoord = DataFrame.iloc[i,1]
        landmark_Ycoord = DataFrame.iloc[i,2]
        
        df = df.append([pix_sampling_training(imageName, image, landmark_Xcoord, landmark_Ycoord, R,Rmax, NposPIX)])
        df.reset_index(drop=True)
    return(df)
'''

def generate_sample_df(Image_Folder, DataFrame, R = 9, Rmax = 300, NposPIX =100):
    #df = pd.DataFrame(columns=['label', 'coord', 'Image_name'])
    images_sampled = {}
    for i in range(1,len(DataFrame)):
        imageName = DataFrame.iloc[i,0]
        image = Image.open(Image_Folder + imageName)
        
        landmark_Xcoord = DataFrame.iloc[i,1]
        landmark_Ycoord = DataFrame.iloc[i,2]
        images_sampled[imageName] = pix_sampling_training(imageName, image, landmark_Xcoord, landmark_Ycoord, R,Rmax, NposPIX)
        #df = df.append([pix_sampling_training(imageName, image, landmark_Xcoord, landmark_Ycoord, R,Rmax, NposPIX)])
        #df.reset_index(drop=True)
    return(images_sampled)


'''
def generate_training_data(df ,Image_Folder, feature_types , feature_coords = None, crop_size = 50, resolution = 5):
    #start_time = datetime.datetime.now()
    window = np.zeros((resolution,resolution))
    features = FeatureExtract(window, feature_types,feature_coords)
    df_features = np.zeros((len(df[list(df.keys())[0]])*len(list(df.keys())),len(features)))
    total = len(df)
    counter = 0
    for i in range(len(df)):
        counter += 1
        imgname = df.iloc[i,2]

        img = Image.open(Image_Folder+ imgname)

        window = get_window(img, df.iloc[i,1], crop_size = 50, resolution = 5)

        features = FeatureExtract(rgb2gray(asarray(window))
                                         , feature_types)

        df_features[i,:] = np.array(features)

 
        df_final = pd.concat([df.reset_index(),pd.DataFrame(df_features)], axis=1)

    return(df_final) 
'''
def generate_training_data(df ,Image_Folder, feature_count ,feature_types , feature_coords = None, crop_size = 50, resolution = 5):
    #start_time = datetime.datetime.now()
    print(feature_count)
    if feature_count == None:
        window = np.zeros((resolution,resolution))
        features = FeatureExtract(window, feature_types,feature_coords)
        df_features = np.zeros((len(df[list(df.keys())[0]])*len(list(df.keys())),len(features)))
    else:
        df_features = np.zeros((len(df[list(df.keys())[0]])*len(list(df.keys())),feature_count))
    total = len(df)
    counter = 0
    df_samples_to_concat = pd.DataFrame(columns=['label', 'coord', 'Image_name'])
    counter_image = 0
    for imgname in df:
        #print(imgname)
        start = datetime.datetime.now()
        df_samples_to_concat = df_samples_to_concat.append([df[imgname]])
        img = Image.open(Image_Folder+ imgname)
        counter_image += 1
        for i in range(len(df[imgname])):
            


            window = get_window(img, df[imgname].iloc[i,1], crop_size = crop_size, resolution = resolution)

            features = FeatureExtract(rgb2gray(asarray(window))
                                            , feature_types, feature_coords)

            df_features[counter,:] = np.array(features)
            counter += 1

        print('image:{}'.format(counter_image))
        print('time taken: {}'.format(datetime.datetime.now() - start))
            
            
    
    df_final = pd.concat([df_samples_to_concat.reset_index(),pd.DataFrame(df_features)], axis=1)
            

    return(df_final)    
    