# App to predict penguin species
# Using user provided data for model training

# Import libraries
import pandas as pd             # Pandas
import streamlit as st          # Streamlit
import matplotlib.pyplot as plt # Matplotlib
import seaborn as sns           # Seaborn

# Module to save and load Python objects to and from files
import pickle 

# Package to implement Decision Tree Model
import sklearn
from sklearn.tree import DecisionTreeClassifier

# Package for data partitioning
from sklearn.model_selection import train_test_split

# Package to calculate f1_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

st.title('Penguin Classifier: A Machine Learning App') 

st.write("This app uses 6 inputs to predict the species of penguin using " 
         "a model built on the Palmer's Penguin's dataset. Use the form below" 
         " to get started!") 

# Asking users to input their own data
penguin_file = st.file_uploader('Upload your own penguin data to train the model') 

# Display an example dataset and prompt the user 
# to submit the data in the required format.
st.write("Please ensure that your data adheres to this specific format:")

# Cache the dataframe so it's only loaded once
@st.cache_data
def load_data(filename):
  df = pd.read_csv(filename)
  return df

data_format = load_data('penguins_sample.csv')
st.dataframe(data_format, hide_index = True)

# Setting the default option to load our Decision Tree model
# if there is no penguin file uploaded by the user
if penguin_file is None: 
    # Load default dataset
    # This dataset will be used later for plotting histograms
    # in case if the user does not provide any data
    penguin_df = pd.read_csv('penguins.csv') 

    # Loading model and mapping pickle files
    dt_pickle = open('decision_tree_penguin.pickle', 'rb') 
    map_pickle = open('output_penguin.pickle', 'rb') 
    clf = pickle.load(dt_pickle) 
    unique_penguin_mapping = pickle.load(map_pickle) 
    dt_pickle.close() 
    map_pickle.close() 

# If the file is provided, we need to clean it and train a model on it
# similar to what we did in the Jupyter notebook
else: 
    # Load dataset as dataframe
    penguin_df = pd.read_csv(penguin_file) 
    # Dropping null values
    penguin_df = penguin_df.dropna() 
    # Output column for prediction
    output = penguin_df['species'] 
    # Input features (excluding year column)
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                           'flipper_length_mm', 'body_mass_g', 'sex']] 
    # One-hot-encoding for categorical variables
    features = pd.get_dummies(features) 
    # Factorize output feature (convert from string to number)
    output, unique_penguin_mapping = pd.factorize(output) 
    # Data partitioning into training and testing 
    train_X, test_X, train_y, test_y = train_test_split(features, output, test_size = 0.2, random_state = 1) 
    # Defining prediction model
    clf = DecisionTreeClassifier(random_state = 0)
    # Fitting model on training data
    clf.fit(train_X, train_y) 
    # Making predictions on test set
    y_pred = clf.predict(test_X) 
    # Calculating F1-score of the model on test set
    score = round(f1_score(y_pred, test_y, average = 'macro'), 2) 
    st.write('We trained a Decision Tree model on these data,' 
             ' it has an F1-score of {}! Use the ' 
             'inputs below to try out the model.'.format(score))

# After creating the model, we need inputs from the user for prediction
# NOTE: This time we make an improvement. In the previous case, each time
# a user changes an input in the app, the entire app reruns.
# We can use st.form() and st.submit_form_button() to wrap the rest of 
# user inputs in and allow the user to change all of the inputs and submit
# the entire form at once instead of multiple times
with st.form('user_inputs'): 
  island = st.selectbox('Penguin Island', options=[
    'Biscoe', 'Dream', 'Torgerson']) 
  sex = st.selectbox('Sex', options=[
    'Female', 'Male']) 
  bill_length = st.number_input(
    'Bill Length (mm)', min_value=0) 
  bill_depth = st.number_input(
    'Bill Depth (mm)', min_value=0) 
  flipper_length = st.number_input(
    'Flipper Length (mm)', min_value=0) 
  body_mass = st.number_input(
    'Body Mass (g)', min_value=0) 
  st.form_submit_button() 

# Putting sex and island variables into the correct format
island_biscoe, island_dream, island_torgerson = 0, 0, 0 
if island == 'Biscoe': 
  island_biscoe = 1 
elif island == 'Dream': 
  island_dream = 1 
elif island == 'Torgerson': 
  island_torgerson = 1 

sex_female, sex_male = 0, 0 
if sex == 'Female': 
  sex_female = 1 
elif sex == 'Male': 
  sex_male = 1 

# Create prediction and display it to user
new_prediction = clf.predict([[bill_length, bill_depth, flipper_length, 
  body_mass, island_biscoe, island_dream, 
  island_torgerson, sex_female, sex_male]]) 

# Map prediction with penguin species
prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting Your Penguin's Species")
st.write('We predict your penguin is of the {} species'.format(prediction_species)) 

# Adding histograms for continuous variables for model explanation
st.write('Below are the histograms for each continuous variable '
         'separated by penguin species. The vertical line '
         'represents your inputted value.')

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_length_mm'], hue = penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_depth_mm'], hue = penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['flipper_length_mm'], hue = penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Bill Flipper by Species')
st.pyplot(ax)

# NOTE: sns.distplot() function accepts the data variable as an argument 
# and returns the plot with the density distribution