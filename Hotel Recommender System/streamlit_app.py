# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import random

import warnings
warnings.filterwarnings("ignore")




# Load the dataset
file_path = 'hotels.csv'
sample_size = 5000  # Adjust the sample size as needed

# Set a random seed for reproducibility
random.seed(42)




df=pd.read_csv(file_path)

hotel_df = df.copy()
users_with_enough_interactions_df = hotel_df.groupby(['userCode']).size().groupby('userCode').size()


users_interactions_count_df = hotel_df.groupby(['userCode','name']).size().groupby('userCode').size()


users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 2].reset_index()[['userCode']]



interactions_from_selected_users_df = hotel_df.merge(users_with_enough_interactions_df,
               how = 'right',
               left_on = 'userCode',
               right_on = 'userCode')



# Encode userCode and hotel name to numeric values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
#df_hotel['userCode'] = label_encoder.fit_transform(df_hotel['userCode'])
interactions_from_selected_users_df['name_encoded'] = label_encoder.fit_transform(interactions_from_selected_users_df['name'])

import math
def smooth_user_preference(x):
    return math.log(1+x, 2)

interactions_full_df = interactions_from_selected_users_df.groupby(['name_encoded','userCode'])['price'].sum().reset_index()


interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                               stratify=interactions_full_df['userCode'],
                                   test_size=0.25,
                                   random_state=42)

x_test=set(interactions_test_df['userCode'])
x_train=set(interactions_train_df['userCode'])

only_in_set1 = x_train - x_test


#print("Elements in train but not in test:", only_in_set1)

only_in_set2 = x_test - x_train

#print("Elements in test but not in train:", only_in_set2)



#Creating a sparse pivot table with users in rows and items in columns
items_users_pivot_matrix_df = interactions_train_df.pivot(index='userCode',
                                                          columns='name_encoded',
                                                          values='price').fillna(0)


items_users_pivot_matrix = items_users_pivot_matrix_df.values
#items_users_pivot_matrix[:10]

user_ids = list(items_users_pivot_matrix_df.index)
#user_ids[:10]

#items_users_pivot_matrix.shape

# The number of factors to factor the item-user matrix.
NUMBER_OF_FACTORS_MF = 8

import scipy
from scipy.sparse.linalg import svds
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(items_users_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)


#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = items_users_pivot_matrix_df.columns,index=user_ids).transpose()
#cf_preds_df.head()

class CFRecommender:

    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df , items_df):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=5, verbose=False):
        if user_id not in self.cf_predictions_df.columns:
            raise KeyError(f"User '{user_id}' not found in prediction data.")
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['name_encoded'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
                
            # Merge recommendations_df with items_df
            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='name_encoded',
                                                          right_on='name_encoded')[['name_encoded','name','recStrength']]
            recommendations_df=pd.DataFrame(recommendations_df.groupby('name').max('recStrength').sort_values('recStrength', ascending=False))

        return recommendations_df

# Assuming cf_preds_df and interactions_from_selected_users_df are defined elsewhere
cf_recommender_model = CFRecommender(cf_preds_df, interactions_from_selected_users_df)

#with open('cf_recommender_model.pkl', 'wb') as f:
    #pickle.dump(cf_recommender_model, f)

#cf_recommender_model.recommend_items(590)

import streamlit as st

def main():
    st.set_page_config(page_title="Hotel Recommendation System", layout="centered")

    # Header Section
    st.title("Hotel Recommendation System")
    st.markdown(
        """
        Welcome to the Hotel Recommendation App! 
        Select your user code below to get personalized hotel recommendations based on your preferences.
        """
    )

    # Sidebar for user input
    st.sidebar.header("User Input")
    usercode_options = hotel_df['userCode'].unique()
    selected_usercode = st.sidebar.selectbox('Select User Code', usercode_options)

    # Button to get recommendations
    st.sidebar.markdown("---")
    st.sidebar.header("Get Recommendations")

    if st.sidebar.button("Recommend Hotels"):
        try:
            recommended_hotels = cf_recommender_model.recommend_items(selected_usercode, verbose=True)
            
            if recommended_hotels.empty:
                st.warning("No recommendations found for the selected user.")
            else:
                st.success(f"Top recommendations for User {selected_usercode}:")
                st.dataframe(recommended_hotels)
        except KeyError:
            st.error(f"User {selected_usercode} does not have enough interaction data to generate recommendations.")

    # Footer Section
    st.markdown("---")
    st.markdown(
        """
        *Powered by Collaborative Filtering Algorithm.*
        
        Developed by souvik using Streamlit.
        """
    )

if __name__ == '__main__':
    main()
