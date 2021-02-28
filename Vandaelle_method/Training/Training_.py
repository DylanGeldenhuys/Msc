
from utilities.utilities import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt



class Train:

    def __init__(self,Image_folder,  DataFrame, modelobject=RandomForestClassifier() , crop_size = 50, resolution = 5, feature_types = ['type-4' , 'type-2-x', 'type-2-y','type-3-x', 'type-3-y'] , report_dir = './'):
        self.modelobject = modelobject
        self.Image_folder = Image_folder
        self.DataFrame = DataFrame
        self.feature_types = feature_types
        self.crop_size = crop_size
        self.report_dir = report_dir
        
        self.resolution = resolution
        self.feature_coords = None
        #self.val_model = modelobject
        self.features_selected = 'no'
    #1
    def sample(self,  R=9,Rmax=300, NposPIX=100):
        self.R = R
        self.Rmax = Rmax
        self.NposPIX = NposPIX
        self.samples = generate_sample_df(Image_Folder= self.Image_folder, DataFrame= self.DataFrame, R =self.R, Rmax = self.Rmax, NposPIX= self.NposPIX)
        
    
    #2 #5
    def extract_haar_features(self, feature_count = None):
        self.feature_data = generate_training_data(df= self.samples, Image_Folder= self.Image_folder, feature_count = feature_count, feature_types = self.feature_types, feature_coords = self.feature_coords ,crop_size= self.crop_size , resolution= self.resolution  )
    #3 #6
    def fit(self):

        self.X = self.feature_data.drop(['label','coord','Image_name', 'index'],  axis=1)
        print(self.X.shape)
        self.y = self.feature_data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.9,shuffle = False, stratify = None)
        self.modelobject.fit(self.X_train, self.y_train)
        print('model fitted')
        if self.features_selected == 'no':
            self.auc_full_features = roc_auc_score(self.y_test, self.modelobject.predict_proba(self.X_test)[:, 1])
            self.classification_report_full_features = classification_report(self.y_test, self.modelobject.predict(self.X_test))
            #file = open('classification_report{}'.format(self.DataFrame.columns[1]), 'w')
            #file.write(self.classification_report_full_features + self)
            #file.close()    
            print(self.classification_report_full_features)
            print('AUC:{}'.format(self.auc_full_features)     )
            #file = open(self.report_dir + 'models_reportclassification_report{}.txt'.format(self.DataFrame.columns[1]), 'w')
            #file.write(self.classification_report_full_features + '\n' + self.auc_full_features)
            #file.close()
        else:
            self.auc_subs_features = roc_auc_score(self.y_test, self.modelobject.predict_proba(self.X_test)[:,1])
            self.classification_report_subs_features = classification_report(self.y_test, self.modelobject.predict(self.X_test)) 
            print(self.classification_report_subs_features)
            print('AUC:{}'.format(self.auc_subs_features)     )


    def feature_select(self):

        self.features_selected = 'yes'
        feature_coord, feature_type = \
                        haar_like_feature_coord(self.resolution, self.resolution,feature_type=feature_types)
        idx_sorted = np.argsort(self.modelobject.feature_importances_)[::-1]
        cdf_feature_importances = np.cumsum(self.modelobject.feature_importances_[idx_sorted])
        cdf_feature_importances /= cdf_feature_importances[-1]
        sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.9)
        sig_feature_percent = round(sig_feature_count /
                            len(cdf_feature_importances) * 100, 1)
        print(('{} features, or {}%, account for 90% of branch points in the '
       'random forest.').format(sig_feature_count, sig_feature_percent))
       # Select the determined number of most informative features
        feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
        feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]
        self.feature_coords = feature_coord_sel
        self.feature_types = feature_type_sel
        self.sig_feature_count = sig_feature_count

    #    pass

