import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Simple Iris Flower Prediction App
This app predicts **Iris flower** type
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('0 - Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('1 - Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('2 - Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('3 - Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

iris_df=pd.DataFrame(X)
iris_df['class']=Y

# Show Entire Dataframe
if st.checkbox("Show DataFrame"):
	st.write(iris_df)

# Show Description
if st.checkbox("Show summary of Dataset"):
	st.write(iris_df.describe())

# Show Plots
# if st.checkbox("Simple Bar Plot with Matplotlib "):
	#iris_df.plot(kind='bar')
	#st.pyplot()

# Show Plots
if st.checkbox("Simple Correlation Plot with Matplotlib "):
    fig, ax = plt.subplots()
    sns.set(font_scale=1)
    sns.heatmap(iris_df.corr(), annot=True, linewidths=.5, ax=ax)
    st.write(fig)

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)