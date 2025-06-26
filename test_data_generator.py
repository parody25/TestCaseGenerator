import streamlit as st
import pandas as pd
import json
import yaml
import openpyxl
from io import StringIO, BytesIO
import openai
from dotenv import load_dotenv
import tempfile
import os
import zipfile
import re

load_dotenv()
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TEST_DATA_RESPONSE_SCHEMA = load_json_file('test_data_response_schema.json')

def test_data_generator():
    st.title("🧪 Test Data Generator")
    st.markdown("Upload your test case schema and generate comprehensive test data")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Schema File", 
        type=["csv", "json", "xlsx", "yaml", "yml"],
        help="Supported formats: CSV, JSON, Excel, YAML"
    )
    
    # Configuration options
    with st.expander("⚙️ Generation Settings"):
        col1, col2 = st.columns(2)
        with col1:
            record_count = st.slider("Records per table", 5, 100, 20)
            include_edge_cases = st.checkbox("Include edge cases", True)
        with col2:
            data_quality = st.select_slider(
                "Data Quality Mix", 
                options=["All Valid", "Mostly Valid", "Balanced", "Mostly Invalid", "All Invalid"],
                value="Balanced"
            )
        business_rules = st.text_area("Additional Business Rules (optional)")
    
    if st.button("✨ Generate Test Data", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a schema file first")
            return
            
        with st.spinner("Processing schema and generating test data..."):
            try:
                # Parse the uploaded file based on its type
                file_ext = uploaded_file.name.split('.')[-1].lower()
                schema_info = parse_schema_file(uploaded_file, file_ext)
                
                # Generate test data using AI
                test_data = generate_test_data(
                    schema_info=schema_info,
                    record_count=record_count,
                    include_edge_cases=include_edge_cases,
                    data_quality=data_quality,
                    business_rules=business_rules
                )
                
                # Display and download options
                display_and_download_results(test_data, file_ext)
                
            except Exception as e:
                st.error(f"Error generating test data: {str(e)}")
                st.stop()

def parse_schema_file(uploaded_file, file_ext):
    """Parse the uploaded schema file based on its format"""
    file_content = uploaded_file.read()
    
    if file_ext in ['csv']:
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        return {"tables": {"main_table": df.to_dict(orient='list')}}
    
    elif file_ext in ['json']:
        return json.loads(file_content)
    
    elif file_ext in ['xlsx']:
        wb = openpyxl.load_workbook(BytesIO(file_content))
        result = {}
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            data = sheet.values
            cols = next(data)
            df = pd.DataFrame(data, columns=cols)
            result[sheet_name] = df.to_dict(orient='list')
        return {"tables": result}
    
    elif file_ext in ['yaml', 'yml']:
        return yaml.safe_load(file_content)
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def generate_test_data(schema_info, record_count, include_edge_cases, data_quality, business_rules):
    """Generate test data using AI based on the schema"""
    prompt = f"""
    You are a test data generator.
    Generate test data EXACTLY matching this schema:

    Schema:
    {json.dumps(schema_info, indent=2)}

    REQUIREMENTS:
    - Generate EXACTLY {record_count} records per table
    - Data quality: {data_quality}
    - Edge cases: {'INCLUDE' if include_edge_cases else 'EXCLUDE'}
    - Business rules: {business_rules if business_rules else 'None'}

    OUTPUT REQUIREMENTS:
    - MUST include populated 'tables' with all specified fields
    - MUST maintain all schema constraints
    - Include '_metadata' with generation details
    - Return ONLY the JSON object (no markdown, no text)
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a sophisticated test data generation system."},
            {"role": "user", "content": prompt}
        ],
        #tools=[{ "type": "function", "function": TEST_DATA_RESPONSE_SCHEMA }],
        #tool_choice={"type": "function", "function": {"name": "generate_test_data"}},
        temperature=0.7,
        max_tokens=4000
    )
    raw_response = response.choices[0].message.content.strip()
    print("Raw response:\n", raw_response)
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    #tool_call = response.choices[0].message.tool_calls[0]
    #if tool_call.function.name == "generate_test_data":
        #return json.loads(tool_call.function.arguments)
    #else:
        #raise ValueError("Unexpected function call response")

def display_and_download_results(test_data, original_format):
    """Display results and provide download options"""
    st.success("✅ Test Data Generated Successfully!")
    
    with st.expander("📊 Preview Generated Data"):
        st.json(test_data)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download as JSON",
            data=json.dumps(test_data, indent=2),
            file_name="test_data.json",
            mime="application/json"
        )

    with col2:
        if "tables" in test_data:
            csv_data = {}
            for table in test_data["tables"]:
                table_name = table.get("table_name", "unnamed_table")
                columns = table.get("columns", [])
                df = pd.DataFrame(columns)
                csv_data[table_name] = df.to_csv(index=False)

            if csv_data:
                if len(csv_data) == 1:
                    st.download_button(
                        label="Download as CSV",
                        data=list(csv_data.values())[0],
                        file_name="test_data.csv",
                        mime="text/csv"
                    )
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_path = os.path.join(tmpdir, "test_data.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for table_name, csv_content in csv_data.items():
                                zipf.writestr(f"{table_name}.csv", csv_content)

                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="Download as CSV (ZIP)",
                                data=f,
                                file_name="test_data.zip",
                                mime="application/zip"
                            )

    with col3:
        if original_format in ['xlsx']:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if "tables" in test_data:
                    for table_name, table_data in test_data["tables"].items():
                        if table_name != "_metadata":
                            pd.DataFrame(table_data).to_excel(
                                writer, 
                                sheet_name=table_name[:31],
                                index=False
                            )
            st.download_button(
                label="Download as Excel",
                data=output.getvalue(),
                file_name="test_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
