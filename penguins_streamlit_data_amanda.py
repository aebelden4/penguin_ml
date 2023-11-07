# Amanda's Version!
# App to predict penguin species
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle

st.title('Penguin Classifier: A Machine Learning App') 

# Display the image
st.image('penguins.png', width = 400)

st.write("Upload your own CSV file to make multiple penguin species predictions or use the form below to make a prediction for one penguin!")
# Adding Streamlit funcitions to get user data file
# Allow user to upload a data file
penguin_file = st.file_uploader("Upload Your Penguins CSV")

if penguin_file is not None:
    penguins_df = pd.read_csv(penguin_file)
    choice = "file"
else:
    choice = "dropbox"



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


# Add if-statement that can allow file or dropboxes 
if choice == "dropbox":
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

    prediction_prob = clf.predict_proba([[bill_length_mm, bill_depth_mm, flipper_length_mm, 
    body_mass_g, island_Biscoe, island_Dream, island_Torgerson, sex_female, sex_male]]) 

    # Map prediction with penguin species
    prediction_species = unique_penguin_mapping[new_prediction][0]

    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.write('We predict your penguin is of the {} species with {:.0%} probability'.format(prediction_species, prediction_prob.max())) 
else:
    # Get variables out of file
    # NOTE: Make sure that variable names are same as that of training dataset
    island = penguins_df["island"]
    sex = penguins_df["sex"]
    bill_length_mm = penguins_df["bill_length_mm"]
    bill_depth_mm = penguins_df["bill_depth_mm"]
    flipper_length_mm = penguins_df["flipper_length_mm"]
    body_mass_g = penguins_df["body_mass_g"]

    # Putting sex and island variables into the correct format
    # so that they can be used by the model for prediction
    n = len(island)
    islands = pd.get_dummies(island)
    sexs = pd.get_dummies(sex)
    #this won't work if user does not have all the categories bc the predictions won't have correct input

    for i in range(n):
        # Using predict() with csv data provided by the user
        new_prediction = clf.predict([[bill_length_mm[i], bill_depth_mm[i], flipper_length_mm[i], 
        body_mass_g[i], islands["Biscoe"][i], islands["Dream"][i], islands["Torgersen"][i], sexs["female"][i], sexs["male"][i]]]) 
        
        predict_prob = clf.predict_proba([[bill_length_mm[i], bill_depth_mm[i], flipper_length_mm[i], 
        body_mass_g[i], islands["Biscoe"][i], islands["Dream"][i], islands["Torgersen"][i], sexs["female"][i], sexs["male"][i]]])

        # Map prediction with penguin species
        prediction_species = unique_penguin_mapping[new_prediction][0]

        # Add predictions to dataframe
        penguins_df.loc[penguins_df.index[i], "Predicted Species"] = prediction_species
        penguins_df.loc[penguins_df.index[i], "Predicted Probability"] = predict_prob.max()



    # Show the predicted species on the app
    st.subheader("Predicting Your Penguin's Species")
    st.dataframe(penguins_df)




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

