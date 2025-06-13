import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import json
from io import BytesIO
import re
import fitz  # PyMuPDF for PDF text extraction
from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import tempfile


load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

def preprocess_brd_text_with_vectorstore(brd_text: str, index_path: str = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(brd_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    if index_path:
        vectorstore.save_local(index_path)
    return vectorstore

def retrieve_relevant_brd_sections(vectorstore, query: str, k: int = 20):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])


def show_logo():
    st.image("Bank_Muscat_logo.png", width=200)

def test_case_generator2():
    show_logo()
    st.markdown("<h1 style='text-align: center;'>GenAI Testcase Generator</h1>", unsafe_allow_html=True)

    model_option = st.selectbox("Select a model", ["llama3", "gpt-4o"])

    upload_mode = st.radio("Choose input type", ["Excel User Stories", "BRD PDF"])
    uploaded_file = None

    if upload_mode == "Excel User Stories":
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    elif upload_mode == "BRD PDF":
        uploaded_file = st.file_uploader("Upload a BRD PDF file", type=["pdf"])

    user_stories_features = []

    if uploaded_file:
        if upload_mode == "Excel User Stories":
            try:
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Select sheet", xls.sheet_names)
                df = pd.read_excel(xls, sheet_name=sheet_name)
                user_stories_features = [" ".join(row.astype(str).values) for _, row in df.iterrows()]
                st.session_state.test_case_generator = user_stories_features
                st.success("Excel file uploaded and processed successfully!")
            except Exception as e:
                st.error(f"Error reading the Excel file: {e}")
        elif upload_mode == "BRD PDF":
            try:
                pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                pdf_text = "\n".join(page.get_text() for page in pdf_doc)
                user_stories_features = [pdf_text]
                st.session_state.test_case_generator = user_stories_features
                st.success("BRD PDF uploaded and text extracted successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # st.subheader("Input Content")
    # if st.session_state.get("test_case_generator"):
    #     for story in st.session_state.test_case_generator:
    #         st.write(story[:1000] + "..." if len(story) > 1000 else story)
    # else:
        # st.info("Please upload a valid Excel or BRD file.")

    if st.session_state.get("test_case_generator"):
        additional_context = st.text_area("Enter additional context (optional)", key="additional_context")

        if st.button('Generate Test Cases'):
            try:
                if upload_mode == "Excel User Stories":
                    system_message = (
                        "You are an assistant designed to create test cases on the following user stories and features:\n\n"
                        + "\n\n".join(f"'''{story}'''" for story in st.session_state.test_case_generator)
                        + f"\n\nAdditional Context: '''{additional_context}'''"
                    )
                elif upload_mode == "BRD PDF":
                    brd_text = st.session_state.test_case_generator[0]
                    with st.spinner("Embedding and retrieving relevant BRD sections..."):
                        vectorstore = preprocess_brd_text_with_vectorstore(brd_text)
                        relevant_brd = retrieve_relevant_brd_sections(
                            vectorstore,
                            query="Extract all user stories, features, and business rules required to generate UAT test cases."
                        )
                        #st.success(relevant_brd)
                    system_message = (
                        "You are a senior QA engineer. Generate a complete set of UAT test cases from the following BRD excerpts. "
                        "Include positive, negative, and corner-case scenarios. Format output as:\n"
                        "{ \"testCases\": [ { \"feature\": \"\", \"userStory\": \"\", \"testScenario\": \"\", \"testCaseTitle\": \"\", "
                        "\"expectedResult\": \"\", \"testSteps\": \"1. Step one 2. Step two\" } ] }\n\n"
                        f"BRD Extract:\n'''{relevant_brd}'''"
                    )


                if model_option == "gpt-4o":
                    completion = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": "Generate the test cases."}
                        ],
                        max_tokens=3000,
                        temperature=0.8
                    )
                    response = completion.choices[0].message.content.strip()
                else:
                    result = llm_llama.generate(
                        prompts=[system_message + "\nGenerate the test cases."],
                        max_tokens=3000,
                        temperature=0.8
                    )
                    response = result.generations[0][0].text.strip()

                st.session_state.generated_test_cases = response
                st.subheader("Generated Test Cases")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating test cases: {e}")

        if st.button("Save to Excel"):
            try:
                response = st.session_state.get("generated_test_cases")
                if not response:
                    st.warning("No generated test cases found.")
                    return

                test_cases_json = extract_json(response)
                if test_cases_json is None:
                    return

                flattened_data = process_test_cases_json(test_cases_json)
                df = pd.DataFrame(flattened_data)

                output = BytesIO()
                df.to_excel(output, index=False)
                output.seek(0)
                st.download_button(label="📥 Download Excel", data=output, file_name="generated_test_cases.xlsx")
            except Exception as e:
                st.error(f"Error saving to Excel: {e}")

# Run app
if __name__ == "__main__":
    test_case_generator2()
