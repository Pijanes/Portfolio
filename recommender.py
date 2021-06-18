'''Berechnung der recommendations'''

# Model (NMF)
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
# Read in model
with open ('nmf_model.bin', 'rb') as file:
    nmf = pickle.load(file)

R=pd.read_csv('R.csv')
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
imputer.fit(R)

movie_title=pd.read_csv('movietitle_db_id.csv')

# user input
def get_recommendations(html_form_data):
    # Step 6-9 jupyter notebook
    # Step 6
    new_user = np.zeros((9724,))
    new_user_df = pd.DataFrame(html_form_data, index=['ratings'], columns=R.columns)
    boolean_mask = new_user_df.isnull().any()
    new_user_df.T[boolean_mask]
    # Step 7
    new_user_df = imputer.transform(new_user_df)
    P_new = pd.DataFrame(data=nmf.transform(new_user_df),
                     columns=['feature1', 'feature2','feature3','feature4','feature5','feature6', 'feature7','feature8','feature9','feature10','feature11', 'feature12','feature13','feature14','feature15','feature16', 'feature17','feature18','feature19','feature20'],
                     index=['new_user'])
    #Step 8
    recommendations = pd.DataFrame(data=np.dot(P_new, nmf.components_),
                               columns=R.columns,
                               index=['new_user'])
    print('AA',recommendations.head(1), recommendations.shape)
    #Step 9
    final_recommendations = recommendations.T[boolean_mask].sort_values('new_user').iloc[-1:]
    print('BB',final_recommendations.head(1), final_recommendations.shape)
    final_recommendations_t=final_recommendations.T
    final_recommendations.reset_index(inplace=True)
    neu_var=int(final_recommendations.iloc[0,0])
    print('ZZ',neu_var)
    
    movie_title_t=movie_title.set_index('movie_id')#.T
    print('CC',final_recommendations,'DD',movie_title_t.shape)
    
    return movie_title_t.iloc[neu_var,0]
