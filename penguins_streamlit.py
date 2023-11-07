# App to predict penguin species
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pickle

st.title('Penguin Classifier: A Machine Learning App') 

# Display the image
st.image('penguins.png', width = 400)

st.write("This app uses 6 inputs to predict the species of penguin using " 
         "a model built on the Palmer's Penguin's dataset. Use the form below" 
         " to get started!") 

# Reading the pickle files that we created before 
dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
map_pickle = open('output_penguin.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
unique_penguin_mapping = pickle.load(map_pickle) 
dt_pickle.close() 
map_pickle.close() 

# Checking if these are the same Python objects that we used before
# st.write(clf)
# st.write(unique_penguin_mapping)

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

# st.write('The user inputs are {}'.format([island, sex, bill_length, bill_depth, flipper_length, body_mass]))

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

# Using predict() with new data provided by the user
new_prediction = clf.predict([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
  body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

# Map prediction with penguin species
prediction_species = unique_penguin_mapping[new_prediction][0]

# Show the predicted species on the app
st.subheader("Predicting Your Penguin's Species")
st.write('We predict your penguin is of the {} species'.format(prediction_species)) 

# Showing Feature Importance plot
st.write('We used a machine learning model (Decision Tree) to '
         'predict the species, the features used in this prediction '
         'are ranked by relative importance below.')
st.image('feature_imp.svg')


