import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import json
from io import BytesIO
import re
import time
from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LLAMA and GPT-4o clients
llm_llama = Ollama(model="llama3")
llm_gpt4o = ChatOpenAI(model="gpt-4o")

def extract_json(response):
    try:
        json_str = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL).group(0)
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError) as e:
        st.error(f"Error parsing JSON: {e}")
        return None

def flatten_json(json_obj, parent_key='', sep='_'):
    items = []
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(flatten_json(item, f'{new_key}{sep}{i}', sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            items.extend(flatten_json(item, f'{parent_key}{sep}{i}', sep=sep).items())
    else:
        items.append((parent_key, json_obj))
    return dict(items)

def process_test_cases_json(test_cases_json):
    if isinstance(test_cases_json, dict) and 'testCases' in test_cases_json:
        return [flatten_json(tc) for tc in test_cases_json['testCases']]
    elif isinstance(test_cases_json, list):
        flattened_data = []
        for feature in test_cases_json:
            if 'Test Scenarios' in feature:
                feature_data = {
                    'Feature': feature.get('Feature'),
                    'User Story': feature.get('User Story')
                }
                for scenario in feature['Test Scenarios']:
                    scenario_data = {**feature_data, **flatten_json(scenario)}
                    flattened_data.append(scenario_data)
            else:
                flattened_data.append(flatten_json(feature))
        return flattened_data
    else:
        st.error("Unexpected JSON format.")
        return []

def show_logo():
    st.image("Bank_Muscat_logo.png", width=200)

def test_case_generator2():
    # Page title
    show_logo()
    st.markdown("<h1 style='text-align: center;'>GenAI Testcase Generator</h1>", unsafe_allow_html=True)
    
    # Model selection
    model_option = st.selectbox("Select a model", ["llama3", "gpt-4o"])

    # Check if Excel file is uploaded, if not, provide option to upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    # Initialize user stories and features
    if "test_case_generator" in st.session_state:
        user_stories_features = st.session_state.test_case_generator
    else:
        user_stories_features = []

    # If Excel file is uploaded, process it
    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", xls.sheet_names)
            df = pd.read_excel(xls, sheet_name=sheet_name)  # Use xls object directly

            # Clear existing session state
            if "test_case_generator" in st.session_state:
                del st.session_state.test_case_generator

            user_stories_features = []  # Clear the local list as well

            # Concatenate all text from all columns into a single string for each row
            for index, row in df.iterrows():
                concatenated_story = " ".join(row.astype(str).values)
                user_stories_features.append(concatenated_story)

            # Update session state with uploaded user stories and features
            st.session_state.test_case_generator = user_stories_features
            st.success("Excel file uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")

    # Display user stories and features
    st.subheader("User Stories and Features")
    if user_stories_features:
        # Display each user story and feature
        for story in user_stories_features:
            st.write(story)
    else:
        st.info("No user stories and features found. Please upload an Excel file or go back to upload in the previous page.")

    # Continue with generating test cases and saving to Excel
    if user_stories_features:
        # Text input for additional context
        additional_context = st.text_area("Enter additional context for the user stories and features", key="additional_context")

        # Trigger button to generate test cases
        if st.button('Generate Test Cases'):
            start_time = time.time()
            # Prepare the system message with context
            system_message = f"You are an assistant designed to create test cases on the following user stories and features:\n\n"
            for story in user_stories_features:
                system_message += f"'''{story}'''\n\n"
            system_message += f"Additional Context: '''{additional_context}'''"

            # Generate test cases using OpenAI
            try:
                if model_option == "gpt-4o":
                    completion = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": "Generate test cases for the above user stories and features."}
                        ],
                        max_tokens=3000,
                        temperature=0.8,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )

                    if completion and completion.choices:
                        response = completion.choices[0].message.content.strip()
                else:
                    # Generate test case using Llama3
                    result = llm_llama.generate(
                        prompts=[system_message + "\n" + "Generate test cases for the given user stories and features."],
                        max_tokens=3000,
                        temperature=0.8,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )
                    response = result.generations[0][0].text.strip()
                end_time = time.time()
                elapsed_time = round(end_time - start_time, 2)
                # Store the generated test cases in session state
                st.session_state.generated_test_cases = response

                # Display the generated test cases
                st.subheader("Generated Test Cases")
                st.write(response)
                st.success(f"Test cases generated in {elapsed_time} seconds.")

            except Exception as e:
                st.error(f"Error generating test cases: {e}")

        # Save generated test cases to Excel
        if st.button("Save to Excel"):
            try:
                # Check if the generated test cases exist in session state
                if "generated_test_cases" in st.session_state:
                    response = st.session_state.generated_test_cases

                    # Extract and parse the JSON response
                    test_cases_json = extract_json(response)
                    if test_cases_json is None:
                        return

                    # Process the JSON data to flatten it
                    flattened_data = process_test_cases_json(test_cases_json)

                    # Create DataFrame
                    df = pd.DataFrame(flattened_data)

                    # Save to Excel
                    file_path = os.path.join(os.path.expanduser("~"), "Downloads", "generated_test_cases.xlsx")
                    df.to_excel(file_path, index=False)

                    st.success(f"Generated test cases saved to Excel successfully! [Download here](file://{file_path})")
                else:
                    st.warning("No generated test cases found. Please generate test cases first.")
            except Exception as e:
                st.error(f"Error saving generated test cases to Excel: {e}")

# Run the app
if __name__ == "__main__":
    test_case_generator2()
