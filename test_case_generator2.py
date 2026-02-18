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

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_llama = Ollama(model="llama3")

def extract_json(response):
    try:
        json_str = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
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
    st.image("TCS_logo.png", width=200)

def call_gpt5(system_message, user_message):
    response = openai_client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_output_tokens=16000
    )
    return response.output_text

def test_case_generator2():
    show_logo()
    st.markdown("<h1 style='text-align: center;'>GenAI Testcase Generator</h1>", unsafe_allow_html=True)

    model_option = st.selectbox("Select a model", ["llama3", "gpt-5.1"])
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if "test_case_generator" in st.session_state:
        user_stories_features = st.session_state.test_case_generator
    else:
        user_stories_features = []

    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select sheet", xls.sheet_names)
            df = pd.read_excel(xls, sheet_name=sheet_name)

            if "test_case_generator" in st.session_state:
                del st.session_state.test_case_generator

            user_stories_features = []

            for index, row in df.iterrows():
                concatenated_story = " ".join(row.astype(str).values)
                user_stories_features.append(concatenated_story)

            st.session_state.test_case_generator = user_stories_features
            st.success("Excel file uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error reading the Excel file: {e}")

    st.subheader("User Stories and Features")
    if user_stories_features:
        for story in user_stories_features:
            st.write(story)
    else:
        st.info("Upload an Excel file to begin.")

    if user_stories_features:
        additional_context = st.text_area("Enter additional context")

        if st.button("Generate Test Cases"):
            start_time = time.time()

            system_message = (
                "You are a senior QA automation engineer. "
                "Generate the required test cases in STRICT JSON format only. "
                "Cover all positive, negative, and edge cases. "
                "Return only JSON. No commentary.\n\n"
            )

            for story in user_stories_features:
                system_message += f"'''{story}'''\n\n"

            system_message += f"Additional Context: '''{additional_context}'''"

            try:
                if model_option == "gpt-5.1":
                    response = call_gpt5(
                        system_message,
                        "Generate as many test cases as possible in JSON format."
                    )
                else:
                    result = llm_llama.generate(
                        prompts=[system_message + "\nGenerate test cases in JSON."],
                        max_tokens=3000
                    )
                    response = result.generations[0][0].text

                # Print the generated test cases to the terminal
                print("Generated Test Cases:")
                print(response)

                # Extract and process the JSON
                test_cases_json = extract_json(response)
                if test_cases_json is not None:
                    flattened_data = process_test_cases_json(test_cases_json)
                    df = pd.DataFrame(flattened_data)

                    # # Save to Excel automatically
                    # file_path = os.path.join(os.path.expanduser("~"), "Downloads", "generated_test_cases.xlsx")
                    # df.to_excel(file_path, index=False)
                    # print(f"Test cases saved to Excel: {file_path}")
                else:
                    print("Failed to extract valid JSON from response.")

                st.session_state.generated_test_cases = response
                st.subheader("Generated Test Cases")
                st.code(response, language="json")

                elapsed = round(time.time() - start_time, 2)
                st.success(f"Generated in {elapsed} seconds")

            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Save to Excel"):
            try:
                if "generated_test_cases" not in st.session_state:
                    st.warning("Generate test cases first.")
                    return

                response = st.session_state.generated_test_cases
                test_cases_json = extract_json(response)

                if test_cases_json is None:
                    return

                flattened_data = process_test_cases_json(test_cases_json)
                df = pd.DataFrame(flattened_data)

                file_path = os.path.join(os.path.expanduser("~"), "Downloads", "generated_test_cases.xlsx")
                df.to_excel(file_path, index=False)

                st.success(f"Saved successfully → {file_path}")

            except Exception as e:
                st.error(f"Save failed: {e}")

if __name__ == "__main__":
    test_case_generator2()
