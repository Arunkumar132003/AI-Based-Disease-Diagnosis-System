import streamlit as st
from unsloth import FastLanguageModel
from pydantic import BaseModel, Field, ValidationError
from PIL import Image
import time
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

def custom_css():
    st.markdown(
        """
        <style>

        body {
            background-color: #f9fafc;
            font-family: 'Roboto', sans-serif;
            color: #444;
            margin: 0;
            padding: 0;
        }

        .title {
            font-size: 3em;
            color: #2c3e50;
            text-align: center;
            font-weight: 700;
            margin-bottom: 30px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        h2 {
            font-size: 1.8em;
            color: #34495e;
            margin-bottom: 15px;
        }

        input, textarea, select {
            font-size: 1.1em;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
            width: 100%;
            box-sizing: border-box;
        }

        textarea {
            min-height: 150px;
        }

        .stFileUploader label {
            font-size: 1.2em;
            font-weight: 500;
            color: #444;
            margin-bottom: 10px;
        }

        [data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: #fff;
            padding: 20px;
        }

        [data-testid="stSidebar"] h2 {
            color: #ecf0f1;
        }

        .stButton button {
            background-color: black;
            color: white;
            font-size: 1em;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease, border 0.3s ease, color 0.3s ease;
        }

        .stButton button:hover {
            background-color: white;
            border: 2px solid black;
            color: black;
        }

        .stRadio label {
            font-size: 1.2em;
            font-weight: 500;
            color: #555;
        }

        footer {
            font-size: 0.9em;
            color: #aaa;
            text-align: center;
            padding: 20px 0;
            margin-top: 30px;
        }

        input, textarea, select, .stButton button {
            transition: all 0.3s ease;
        }

        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #6a82fb;
            box-shadow: 0 0 5px rgba(106, 130, 251, 0.5);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

class UserInput(BaseModel):
    age: int
    gender: str
    symptoms: str

@st.cache_resource
def load_model():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    model_name = "iamak132003/disease_diagnosis"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

def save_temp_file(uploaded_file):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def delete_file(file_name):
    try:

        if os.path.exists(file_name):
            os.remove(file_name)
            return f"File '{file_name}' has been deleted."
        else:
            return f"File '{file_name}' does not exist."
    except Exception as e:
        return f"An error occurred while deleting the file: {e}"

def home():
    st.markdown('<h1 class="title">Medical Assistant App</h1>', unsafe_allow_html=True)
    st.write("<h2>Your AI-powered health companion. Choose a feature to begin.</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Lab Report Summarizer"):
            st.session_state.page = "Lab Report Summarizer"
            st.rerun()
    with col2:
        if st.button("Prescription Generator"):
            st.session_state.page = "Prescription Generator"
            st.rerun()
    with col3:
        if st.button("AI Disease Diagnoser"):
            st.session_state.page = "AI Disease Diagnoser"
            st.rerun()


###########################################################
#                  GEMINI API                             #
###########################################################

def generate_summary_from_image(image, prompt):
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([prompt, image])
    return response.text

def lab_report_summarizer():
    st.markdown('<h1 class="title">Lab Report Summarizer</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF/Image", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        file_path = save_temp_file(uploaded_file)
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            img = Image.open(file_path)
            if st.button('Analyze'):
                st.markdown("#### Analyzing the uploaded image...")
                lab= """
                You are an advanced AI assistant designed to provide insightful analysis and accurate lab report analysis based on user-provided documents. Your goal is to deliver clear, reliable, and actionable outputs that are personalized and easy to understand.

                Your scope of assistance includes:
                Lab Report Details Analysis- Extract insights from uploaded lab reports, highlight abnormalities, and provide recommendations.

                SOP for Lab Report Details Analysis:
                1.Input Expectation: The user uploads a lab report in text, image, or scanned format. Extract relevant details such as test names, values, reference ranges, and any remarks.
                2.Processing Steps:
                - Parse and analyze the uploaded report.
                - Identify abnormalities by comparing reported values against standard reference ranges.
                - Provide detailed insights for each abnormal value, including potential causes and associated conditions.
                - If values are normal, affirm their normalcy with brief explanations.
                3.Output Requirements:
                A structured summary of the report, including:
                - Test name.
                - Reported value and reference range.
                - Status: Normal/Abnormal.
                - Detailed explanation of abnormal findings.
                - General health insights based on the overall report.
                - Recommendations for further actions, if applicable (e.g., consulting a doctor, lifestyle changes, retesting).

                Additional Guidelines:
                - NOTE: STRICTLY Avoid any unwanted information. Provide the response specifically for the corresponding tasks without any additional or unrelated content.
                - Avoid mentioning the task details, Provide ONLY the response for the task you're doing.
                """
                response = generate_summary_from_image(img, lab)
                st.markdown(response)
        delete_file(file_path)
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
        st.rerun()


def prescription_exporter():
    st.markdown('<h1 class="title">Prescription Analyzer</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_path = save_temp_file(uploaded_file)
        img = Image.open(file_path)
        prescription= """
            You are an advanced AI assistant designed to provide insightful analysis and accurate prescription generation based on user-provided documents. Your goal is to deliver clear, reliable, and actionable outputs that are personalized and easy to understand.
            Your scope of assistance includes:
            Prescription Generation Based on Uploaded Prescription- Generate detailed prescriptions from uploaded doctor prescriptions with clear instructions.

            SOP for Prescription Generation Based on Uploaded Prescription:
            1.Input Expectation: The user uploads a doctor's prescription in text, image, or scanned format. Extract details such as medication names, dosages, frequency, and additional instructions.
            2.Processing Steps:
            - Parse the prescription to identify all medications.
            - Provide the exact details of each medication, including:
            - Name of the pill.
            - Dosage and frequency.
            - Purpose or use of the medication.
            - Special instructions (e.g., take before/after meals, avoid certain activities).
            - Validate that the prescription aligns with standard treatment protocols (if applicable).
            3.Output Requirements:
            - A structured prescription summary, including:
            - Medication name.
            - Dosage.
            - Timing and frequency.
            - Use/purpose in layman’s terms.
            - Safety instructions or additional advice if needed.

            Additional Guidelines:
             - NOTE: STRICTLY Avoid any unwanted information. Provide the response specifically for the corresponding tasks without any additional or unrelated content.
             - Avoid mentioning the task details, Provide ONLY the response for the task you're doing.
            """
        try:
            summary = generate_summary_from_image(img, prescription)
            st.markdown(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
        delete_file(file_path)
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
        st.rerun()


##############################################################
#                    FINE TUNED MODEL                        #
##############################################################

def ai_disease_diagnosis_loader():
    msg = st.toast('Gathering symptoms...', icon="🔍")
    time.sleep(5)
    msg.toast('Analyzing data...', icon="🧠")
    time.sleep(5)
    msg.toast('Diagnosing...', icon="⚡")
    time.sleep(5)
    msg.toast('Finalizing results...', icon="🔬")
    time.sleep(5)
    msg.toast('Diagnosis ready!', icon="🩺")

def ai_symptoms_predictor():
    st.markdown('<h1 class="title">AI Disease Diagnoser</h1>', unsafe_allow_html=True)
    age = st.number_input("Enter Age:", min_value=5, max_value=100, step=1)
    gender = st.selectbox("Select Gender:", ["Male", "Female", "Other"])
    symptoms = st.text_area("Enter your symptoms")

    if st.button("Predict"):
        try:
            user_input = UserInput(age=age, gender=gender, symptoms=symptoms)
            input_data = f"Symptoms: {user_input.symptoms}\nGender: {user_input.gender}\nAge: {user_input.age}"

            disease_diagnosis_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Generate a response that appropriately completes the request.
            ### Instruction:
            Provide a diagnosis and all the recommendations based on the patient's symptoms.

            ### Input:
            {}

            ### Response:
            {}
            """.format(input_data, "")

            inputs = tokenizer([disease_diagnosis_prompt], return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            ai_disease_diagnosis_loader()

            outputs = model.generate(**inputs, max_new_tokens=800, use_cache=True)
            result = tokenizer.batch_decode(outputs)[0]
            response_only = (
                result.replace("<|begin_of_text|>", "")
                .replace("<|end_of_text|>", "")
                .strip()
            )

            if "### Response:" in response_only:
                diagnosis = response_only.split("### Response:")[1].strip()
                formatted_diagnosis = diagnosis.replace("**", "<strong>").replace("**", "</strong>")
                formatted_diagnosis = formatted_diagnosis.replace("<strong><strong>", "<strong>").replace("</strong></strong>", "</strong>")
                formatted_diagnosis = formatted_diagnosis.replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 15px; background-color: #f9f9f9;">
                        {formatted_diagnosis}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("### Unable to process the response.")

        except ValidationError as e:
            st.error(f"Validation Error: {e}")

    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
        st.rerun()

def main():
    custom_css()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Lab Report Summarizer":
        lab_report_summarizer()
    elif st.session_state.page == "Prescription Generator":
        prescription_exporter()
    elif st.session_state.page == "AI Disease Diagnoser":
        ai_symptoms_predictor()

if __name__ == "__main__":
    main()