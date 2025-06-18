import streamlit as st
import os
import tempfile
from openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv

load_dotenv()

def brd_reader():
    st.title("📄 BRD-Based UAT Test Case Generator")
    uploaded_file = st.file_uploader("Upload a BRD PDF", type=["pdf"])

    if uploaded_file:
        application_id = st.text_input("Enter Application ID", value="brd-app-008")

        if st.button("Generate Test Cases"):
            with st.spinner("🔍 Parsing and embedding your BRD..."):
                file_name = uploaded_file.name
                print(file_name)
                file_bytes = uploaded_file.read()  # Make sure this is read only once
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

                        #st.info(f"📄 Parsed content preview:\n\n{documents[0].get_content()[:1000]}...")

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

                    retriever = vector_db.as_retriever(similarity_top_k=20)
                    query = "What features, workflows, user stories, or rules can be used to generate UAT test cases?"
                    docs = retriever.retrieve(query)
                    context = "\n\n".join([doc.text for doc in docs])

                    st.subheader("📋 Retrieved BRD Context")
                    if not context.strip():
                        st.warning("⚠️ No relevant content found from BRD. Please verify document formatting or try a different query.")
                        return
                    st.code(context[:1500] + "..." if len(context) > 1500 else context)

                    prompt = f"""
You are a senior QA engineer. Your task is to read and understand the attached Business Requirements Document (BRD) and generate a complete set of test cases for User Acceptance Testing (UAT).
Please perform the following:
1) Ingest and analyze the BRD to understand the full scope of the project, including all features, workflows, user roles, and business rules.
2) Generate test cases that cover:
- Positive scenarios: Valid inputs and expected successful outcomes.
- Negative scenarios: Invalid inputs, error handling, and edge cases.
- Corner cases: Boundary conditions, unusual but valid inputs, and performance-related edge cases.
3) Organize test cases by feature/module, and include:
- Test case ID
- Title/Description
- Preconditions
- Test steps
- Expected results
- Priority (High/Medium/Low)
- Type (Positive/Negative/Corner)
4) Ensure the test cases:
- Cover all user roles and permissions
- Validate UI/UX elements if applicable
- Include integration points with other systems or APIs
- Reflect business rules and logic
- Are suitable for manual execution during UAT
5) Provide a summary table of test coverage by feature/module.
Format the output in a structured way (e.g., Markdown table or spreadsheet-ready format). Ensure clarity, completeness, and traceability to the BRD

BRD Content:
\"\"\"{context}\"\"\"
                    """

                    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    completion = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": "Generate the test cases in elaborate."}
                        ],
                        max_tokens=7000,
                        temperature=0.7
                    )

                    response = completion.choices[0].message.content.strip()
                    st.subheader("✅ Generated UAT Test Cases")
                    #st.code(response, language="json")
                    st.code(response)

                except Exception as e:
                    st.error(f"❌ Error during test case generation: {e}")
