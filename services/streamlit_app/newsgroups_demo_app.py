import streamlit as st
from newsgroups_classifier import NewsgroupsClassifier

st.title('Newsgroups Posts Classification Demo')

st.markdown('This is a **Demo Stand**')
st.markdown('Some instructions')

with st.form('form'):
	text_input = st.text_area('Enter news text here: ', 'empty news')
	submit_button = st.form_submit_button('Predict Group')

	if submit_button:
		model = NewsgroupsClassifier()
		label = model.get_topic(text_input)
		st.write('Predicted label: ', f'{label}')