import streamlit as st
import os
import tempfile
import openai
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

TEST_CASE_SCHEMA = load_json_file('test_case_schema.json')
EXAMPLE_TEST_CASE = load_json_file('example_test_case.json')

def brd_reader():
    st.title("📄 BRD-Based UAT Test Case Generator")
    uploaded_file = st.file_uploader("Upload a BRD PDF", type=["pdf"])

    if uploaded_file:
        application_id = st.text_input("Enter Application ID", value="brd-app-008")

        if st.button("Generate Test Cases"):
            with st.spinner("🔍 Parsing and embedding your BRD..."):
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                document_name, ext = os.path.splitext(file_name)
                ext = ext if ext else ".pdf"

                embedding_dir = f"embeddings/{application_id}"
                os.makedirs(embedding_dir, exist_ok=True)
                embedding_path = f"{embedding_dir}/{document_name}_embedding.pkl"

                if not os.path.exists(embedding_path):
                    try:
                        llama_cloud_api = os.getenv("LLAMA_CLOUD_API_KEY")
                        llama_parser = LlamaParse(result_type="markdown", api_key=llama_cloud_api)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        documents = llama_parser.load_data(tmp_path)

                        if not documents:
                            st.error("❌ LlamaParse returned no documents. The BRD may be scanned or empty.")
                            return

                        llm = LlamaOpenAI(model="gpt-4o")
                        parser = MarkdownElementNodeParser(llm=llm, num_workers=10)
                        nodes = parser.get_nodes_from_documents(documents)
                        base_nodes, objects = parser.get_nodes_and_objects(nodes)

                        if not base_nodes and not objects:
                            st.error("❌ No structured content found. The BRD may lack headings or be poorly formatted.")
                            return

                        total_chars = sum(len(n.text) for n in base_nodes + objects)
                        st.info(f"📦 Total extracted content size: {total_chars} characters")

                        index = VectorStoreIndex(base_nodes + objects)
                        index.storage_context.persist(embedding_path)

                        st.success("✅ Embedding created successfully.")
                    except Exception as e:
                        st.error(f"❌ Failed to create embeddings: {e}")
                        return
                else:
                    st.info("✅ Reusing existing embedding.")

            with st.spinner("🤖 Retrieving relevant BRD sections and generating test cases..."):
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
                    vector_db = load_index_from_storage(storage_context=storage_context)

                    retriever = vector_db.as_retriever(similarity_top_k=30)
                    query = "Extract all features, requirements, user stories, workflows, business rules, acceptance criteria, system interactions, and edge cases that should be considered for test case generation. Include both functional and non-functional requirements."
                    docs = retriever.retrieve(query)
                    context = "\n\n".join([doc.text for doc in docs])

                    st.subheader("📋 Retrieved BRD Context")
                    if not context.strip():
                        st.warning("⚠️ No relevant content found from BRD. Please verify document formatting or try a different query.")
                        return
                    st.code(context[:2500] + "..." if len(context) > 2500 else context)

                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    tool_schema = {
                        "name": "generate_test_cases",
                        "description": "Generate structured UAT test cases based on BRD content",
                        "parameters": TEST_CASE_SCHEMA
                    }

                    prompt = f"""
You are a Senior QA Architect.

Generate at least 100 test cases covering:
- Functional and non-functional requirements
- Positive, negative, and edge cases
- Grouped by feature/module
- Traceable to BRD inputs

Refer to this BRD content:
\"\"\"{context}\"\"\"
"""

                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a structured test case generation assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        tools=[{
                            "type": "function",
                            "function": tool_schema
                        }],
                        tool_choice={"type": "function", "function": {"name": "generate_test_cases"}},
                        temperature=0.5
                    )

                    response_json = completion.choices[0].message.tool_calls[0].function.arguments
                    parsed_output = json.loads(response_json)

                    st.subheader("✅ Generated UAT Test Cases")
                    st.code(json.dumps(parsed_output, indent=2), language="json")

                    # JSON download button
                    # st.download_button(
                    #     label="⬇️ Download JSON",
                    #     data=json.dumps(parsed_output, indent=2),
                    #     file_name="uat_test_cases.json",
                    #     mime="application/json"
                    # )

                    # Convert to DataFrame for Excel
                    excel_rows = []
                    suite = parsed_output["test_suite"]
                    for feature in suite["features"]:
                        for case in feature["test_cases"]:
                            excel_rows.append({
                                "Application Name": suite.get("application_name", ""),
                                "BRD Reference": suite.get("brd_reference", ""),
                                "Feature Name": feature.get("feature_name", ""),
                                "Feature Description": feature.get("feature_description", ""),
                                "Test Case ID": case.get("test_case_id", ""),
                                "Title": case.get("title", ""),
                                "Description": case.get("description", ""),
                                "Test Type": case.get("test_type", ""),
                                "Priority": case.get("priority", ""),
                                "Preconditions": case.get("preconditions", ""),
                                "Test Steps": "\n".join(case.get("test_steps", [])),
                                "Expected Result": case.get("expected_result", ""),
                                "Requirements Covered": ", ".join(case.get("requirements_covered", []))
                            })

                    df = pd.DataFrame(excel_rows)

                    # Save Excel to buffer
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False, sheet_name="TestCases")
                    excel_buffer.seek(0)

                    # Excel download button
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=excel_buffer,
                        file_name="uat_test_cases.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f"❌ Error during test case generation: {e}")
