 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('income_check')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('income.jpg')
    image_office = Image.open('income1.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict wether a given adult makes more than $50,000  or not')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("customer will make $50,000 or not?")
    if add_selectbox == 'Online':
        age=st.number_input('age' , min_value=17, max_value=90, value=17)
        capital_gain =st.number_input('capital_gain',min_value=0, max_value=99999, value=0)
        capital_loss = st.number_input('capital_loss', min_value=0, max_value=4356, value=0)
        hours_per_week = st.number_input('hours_per_week', min_value=1, max_value=99, value=1)
      
        workclass = st.selectbox('workclass ', ['Private','Self-emp-not-inc','Local-gov',"State-gov","Self-emp-inc","Other_values"])
        education= st.selectbox('education', ['HS-grad', 'Some-college','Bachelors','Masters','Assoc-voc','Other values'])
        marital_status= st.selectbox('marital_status', ['Married-civ-spouse', 'Never-married','Divorced','Separated	','Widowed','Other values'])
        occupation = st.selectbox('occupation', ['Prof-specialty', 'Craft-repair','Exec-managerial','Adm-clerical','Sales','Other values'])
        relationship = st.selectbox('relationship ', ['Husband	', 'Not-in-family','Own-child','Unmarried','Wife'])
        race = st.selectbox('race', ['White', 'Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'])
        native_country = st.selectbox('native_country', ['United-States','Mexico','Philippines','Germany','Canada','Other values'])
        income_greater_thanornot  = st.selectbox('income_greater_thanornot ', ['yes', 'no'])
        sex=st.selectbox('sex ', ['Male', 'Female'])
      

        output=""
        input_dict={'age':age,'capital_gain':capital_gain,'capital_loss':capital_loss,'sex':sex,'hours_per_week': hours_per_week,'workclass': workclass,'education': education,'marital_status' : marital_status,'occupation':occupation,'relationship':relationship,'race':race,'native_country':native_country,'income_greater_thanornot':income_greater_thanornot}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
            if output == '1':
              output="Adult will earn more than 50kdollar"
            else:
              output="Adult will not earn 50kdollar"  
        st.success('output -- {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()

