# Actual answer
# App to predict penguin species
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Penguin Classifier: A Machine Learning App') 

# Display the image
st.image('penguins.png', width = 400)

st.write("This app uses 6 inputs to predict the species of penguin using " 
         "a model built on the Palmer's Penguin's dataset. Use the following form or upload your dataset" 
         " to get started!") 

# Reading the pickle files that we created before 
dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
map_pickle = open('output_penguin.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
unique_penguin_mapping = pickle.load(map_pickle) 
dt_pickle.close() 
map_pickle.close()

# Option 1: Asking users to input their data as a file
penguin_file = st.file_uploader('Upload your own penguin data')

# Option 2: Asking users to input their data using a form
# Adding Streamlit functions to get user input
# For categorical variables, using selectbox
island = st.selectbox('Penguin Island', options = ['Biscoe', 'Dream', 'Torgerson']) 
sex = st.selectbox('Sex', options = ['Female', 'Male']) 

# For numerical variables, using number_input
# NOTE: Make sure that variable names are same as that of training dataset
bill_length_mm = st.number_input('Bill Length (mm)', min_value = 0) 
bill_depth_mm = st.number_input('Bill Depth (mm)', min_value = 0) 
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value = 0) 
body_mass_g = st.number_input('Body Mass (g)', min_value = 0) 

# Putting sex and island variables into the correct format
# so that they can be used by the model for prediction
island_Biscoe, island_Dream, island_Torgerson = 0, 0, 0 
if island == 'Biscoe': 
   island_Biscoe = 1 
elif island == 'Dream': 
   island_Dream = 1 
elif island == 'Torgerson': 
   island_Torgerson = 1 

sex_female, sex_male = 0, 0 
if sex == 'Female': 
   sex_female = 1 
elif sex == 'Male': 
   sex_male = 1 

# If no file is provided, then allow user to provide inputs using the form
if penguin_file is None:

    # Using predict() with new data provided by the user
    new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
    body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

    new_prediction_prob = clf.predict_proba([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
    body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]])

    # Map prediction with penguin species
    prediction_species = unique_penguin_mapping[new_prediction][0]

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.write('We predict your penguin is of the {} species with {:.0%} probability'.format(prediction_species, new_prediction_prob.max())) 

else:
   # Loading data
   user_df = pd.read_csv(penguin_file) # User provided data
   original_df = pd.read_csv('penguins.csv') # Original data to create ML model
   
   # Dropping null values
   user_df = user_df.dropna() 
   original_df = original_df.dropna() 
   
   # Remove output (species) and year columns from original data
   original_df = original_df.drop(columns = ['species', 'year'])
   # Remove year column from user data
   user_df = user_df.drop(columns = ['year'])
   
   # Ensure the order of columns in user data is in the same order as that of original data
   user_df = user_df[original_df.columns]

   # Concatenate two dataframes together along rows (axis = 0)
   combined_df = pd.concat([original_df, user_df], axis = 0)

   # Number of rows in original dataframe
   original_rows = original_df.shape[0]

   # Create dummies for the combined dataframe
   combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
   original_df_encoded = combined_df_encoded[:original_rows]
   user_df_encoded = combined_df_encoded[original_rows:]

   # Predictions for user data
   user_pred = clf.predict(user_df_encoded)

   # Predicted species
   user_pred_species = unique_penguin_mapping[user_pred]

   # Adding predicted species to user dataframe
   user_df['Predicted Species'] = user_pred_species

   # Prediction Probabilities
   user_pred_prob = clf.predict_proba(user_df_encoded)
   # Storing the maximum prob. (prob. of predicted species) in a new column
   user_df['Predicted Species Prob.'] = user_pred_prob.max(axis = 1)
   
   # Show the predicted species on the app
   st.subheader("Predicting Your Penguin's Species")
   st.dataframe(user_df)

# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

with tab1:
  st.image('dt_visual.svg')
with tab2:
  st.image('feature_imp.svg')
with tab3:
  st.image('confusion_mat.svg')
with tab4:
    df = pd.read_csv('class_report.csv', index_col=0)
    st.dataframe(df)