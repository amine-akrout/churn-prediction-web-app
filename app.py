from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('model')

def predict(model, input_df):
	predictions_df = predict_model(estimator=model, data=input_df)
	predictions = predictions_df['Label'][0]
	return predictions


def main():
	from PIL import Image
	image = Image.open('images/icone.jpg')
	image2 = Image.open('images/image.png')
	st.image(image,use_column_width=False)
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict House prices')
	st.sidebar.image(image2)
	st.title("Predicting house price")
	if add_selectbox == 'Online':
		state =st.selectbox('letter code of the US state of customer residence :',['10 Very Excellent', '9 Excellent','8 Very Good','7 Good' ,'6 Above Average','5 Average','4 Below Average','3 Fair','2 Poor','1 Very Poor'])
		account_length=st.number_input('Number of months the customer has been with the current telco provider :' , min_value=1300, max_value=21600, value=10000)
		output=""
		input_dict={'state':state,'account_length':account_length}
		input_df = pd.DataFrame([input_dict])
		if st.button("Predict"):
			output = predict(model=model, input_df=input_df)
			output = str(output)
		st.success('The estimated parice is : {} $'.format(output))
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			predictions = predict_model(estimator=model,data=data)
			st.write(predictions)
if __name__ == '__main__':
	main()