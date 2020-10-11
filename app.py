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
		state =st.selectbox('letter code of the US state of customer residence :',['','AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA','ID',\
		'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV',\
		'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV','WY'])
		account_length=st.number_input('Number of months the customer has been with the current telco provider :' , min_value=1300, max_value=21600, value=10000)
		area_code=st.selectbox('"area_code_AAA" where AAA = 3 digit area code :' , ['','area_code_408', 'area_code_415', 'area_code_510'])
		international_plan=st.selectbox('The customer has international plan :' , ['','yes', 'no'])
		voice_mail_plan=st.selectbox('The customer has voice mail plan :' , ['','yes', 'no'])
		number_vmail_messages=st.number_input('Number of voice-mail messages. :' , min_value=1300, max_value=21600, value=10000)
		total_day_minutes=st.number_input('Total minutes of day calls :' , min_value=1300, max_value=21600, value=10000)
		total_day_calls=st.number_input('Total minutes of day calls :' , min_value=1300, max_value=21600, value=10000)
		total_day_charge=st.number_input('Total charge of day calls :' , min_value=1300, max_value=21600, value=10000)
		total_eve_minutes=st.number_input('Total minutes of evening calls :' , min_value=1300, max_value=21600, value=10000)
		total_eve_calls=st.number_input('Total number of evening calls :' , min_value=1300, max_value=21600, value=10000)
		total_eve_charge=st.number_input('Total charge of evening calls :' , min_value=1300, max_value=21600, value=10000)
		total_night_minutes=st.number_input('Total minutes of night calls :' , min_value=1300, max_value=21600, value=10000)
		total_night_calls=st.number_input('Total number of night calls :' , min_value=1300, max_value=21600, value=10000)
		total_night_charge=st.number_input('Total charge of night calls :' , min_value=1300, max_value=21600, value=10000)
		total_intl_minutes=st.number_input('Total minutes of international calls :' , min_value=1300, max_value=21600, value=10000)
		total_intl_calls=st.number_input('Total number of international calls :' , min_value=1300, max_value=21600, value=10000)
		total_intl_charge=st.number_input('Total charge of international calls :' , min_value=1300, max_value=21600, value=10000)
		number_customer_service_calls=st.number_input('Number of calls to customer service :' , min_value=1300, max_value=21600, value=10000)
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