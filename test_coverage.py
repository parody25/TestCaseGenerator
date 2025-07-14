import streamlit as st
import pandas as pd
import json
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = LlamaOpenAI(model="gpt-4o")

def test_coverage():
    st.title("🧪 Test Coverage Validator / Gap Detector")

    mode = st.radio("Select Source for Requirements:", ["BRD (with embeddings)", "User Story Excel"])

    if mode == "BRD (with embeddings)":
        application_id = st.text_input("Enter Application ID used during BRD test case generation")

        test_case_file = st.file_uploader("Upload Generated Test Cases (Excel or JSON from BRD)", type=["json", "xlsx"])
        if st.button("🧠 Validate Coverage from BRD"):
            if not application_id or not test_case_file:
                st.warning("Please provide both Application ID and test case file.")
                return

            with st.spinner("Loading BRD embedding and validating test coverage..."):
                try:
                    embedding_dir = f"embeddings/{application_id}"
                    all_files = os.listdir(embedding_dir)
                    pkl_files = [f for f in all_files if f.endswith("_embedding.pkl")]

                    if not pkl_files:
                        st.error("❌ No embedding file found for this Application ID. Please generate BRD test cases first.")
                        return

                    embedding_path = os.path.join(embedding_dir, pkl_files[0])

                    storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                    vector_db = load_index_from_storage(storage_context=storage_context)
                    retriever = vector_db.as_retriever(similarity_top_k=30)

                    context_docs = retriever.retrieve("List all requirements, features, user interactions and edge cases")
                    context = "\n\n".join([doc.text for doc in context_docs])

                    if test_case_file.name.endswith(".json"):
                        test_case_data = json.load(test_case_file)
                        test_case_text = json.dumps(test_case_data, indent=2)[:3000]
                    else:
                        df = pd.read_excel(test_case_file)
                        test_case_text = df.to_csv(index=False)[:3000]

                    prompt = f"""
You are a QA lead. Given the original requirements below (from BRD) and the test cases, identify what is covered and what is missing.

Requirements:
{context[:3000]}

Test Cases:
{test_case_text}

Return a summary of:
1. What requirements are fully covered
2. What is partially covered
3. What is missing entirely
4. A coverage percentage estimation
5. Suggest additional test cases if needed
"""

                    completion = llm.complete(prompt)
                    st.subheader("📋 Coverage Report")
                    st.text_area("Coverage Report (copy below)", value=completion, height=300)
                    st.markdown(completion)

                except Exception as e:
                    st.error(f"❌ Failed to load BRD embedding or process coverage: {e}")

    elif mode == "User Story Excel":
        excel_file = st.file_uploader("Upload User Story Excel", type=["xlsx"])
        test_case_file = st.file_uploader("Upload Test Cases (Excel or JSON from Excel)", type=["json", "xlsx"])

        if st.button("📊 Validate Coverage from Excel"):
            if not excel_file or not test_case_file:
                st.warning("Please upload both the user story Excel and the test case file.")
                return

            with st.spinner("Reading Excel and analyzing coverage..."):
                try:
                    df = pd.read_excel(excel_file)
                    user_stories_text = "\n".join([f"{row['ID']}: {row['Description']}" for _, row in df.iterrows() if 'Description' in row])

                    if test_case_file.name.endswith(".json"):
                        test_case_data = json.load(test_case_file)
                        test_case_text = json.dumps(test_case_data, indent=2)[:3000]
                    else:
                        test_case_df = pd.read_excel(test_case_file)
                        test_case_text = test_case_df.to_csv(index=False)[:3000]

                    prompt = f"""
You are a QA analyst. Compare the following user stories and their test cases.

User Stories:
{user_stories_text[:3000]}

Test Cases:
{test_case_text}

Identify:
- Which user stories are fully tested
- Which are partially tested
- Which are not tested at all
- Calculate approximate test coverage %
- Recommend test cases for gaps
"""

                    completion = llm.complete(prompt)
                    st.subheader("📋 Coverage Report")
                    st.text_area("Coverage Report (copy below)", value=completion, height=300)
                    st.markdown(completion)

                except Exception as e:
                    st.error(f"❌ Failed to analyze Excel-based coverage: {e}")
