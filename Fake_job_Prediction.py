import streamlit as st
import pickle
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from PIL import Image


im=Image.open("Fake Job.png")
st.image(im)



model=pickle.load(open('model.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))
tfidf=pickle.load(open('vect.pkl','rb'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text


def main():

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🕵️ Fake Job Detection System</h1>",unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Check whether a job posting is real or fake using AI</p>",unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.write("Enter the details of the Job")


    #input
    title=st.text_input("Job Title",placeholder="Enter job title...")
    description=st.text_area("Job Description",placeholder="Enter detailed job description...")
    requirements=st.text_area("Job Requirements",placeholder="Enter job requirements...")
    company_profile=st.text_area("Company Profile",placeholder="Enter company profile...")
    benefits=st.text_area("Job Benefits",placeholder="Enter job benefits...")

    employment_type=st.selectbox("Employment Type",["Other","Full-time","unknown","Part-time","Contract","Temporary"])
    required_experience=st.selectbox("Required Experience",["Internship","Not Applicable","unknown","Mid-Senior level","Associate","Entry level","Executive","Director"])
    required_education=st.selectbox("Required Education",["unknown","Bachelor's Degree", "Master's Degree","High School or equivalent","Unspecified","Some College Coursework Completed","Vocational","Certification","Associate Degree","Professional","Doctorate","Some High School Coursework","Vocational - Degree","Vocational - HS Diploma"])
    industry=st.text_input("Industry",placeholder="Enter industry...")
    function=st.text_input("Function",placeholder="Enter function...")

    telecommuting=st.checkbox("Telecommuting")
    has_company_logo=st.checkbox("Has Company logo")
    has_questions=st.checkbox("Has Questions")

    if st.button("Check Job Authenticity"):
        if title.strip() == "" and description.strip() == "":
            st.warning("⚠️ Please enter at least job title or description")
            return
        text=title+" "+company_profile+" "+description+" "+requirements+" "+benefits
        text=clean_text(text)
        text_vect=tfidf.transform([text])

        cat_data=np.array([[employment_type,required_experience,required_education,industry,function]])
        cat_vect=encoder.transform(cat_data)

        num_data=np.array([[int(telecommuting),int(has_company_logo),int(has_questions)]])
        num_vect=csr_matrix(num_data)

        finput=hstack([text_vect,cat_vect,num_vect]).tocsr()

        prediction=model.predict(finput)[0]

        if prediction==1:
            st.error("🚨This job is likely FAKE.Please be cautious.")
        else:
            st.success("✅ This job appears to be REAL")



main()

with st.expander("ℹ️ What does this app do?"):
    st.write("It analyzes job descriptions and predicts if they are fake.")

with st.expander("ℹ️ How can I use this app?"):
    st.write("1. Enter the details of the job you want to check.")
    st.write("2. Click the 'Check Job Authenticity' button.")
    st.write("3. The app will predict if the job is likely to be fake or real.")