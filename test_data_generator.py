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


def format_sql_value(value):
    """Safely format values for SQL insert"""
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (dict, list)):
        # Convert nested dict/list to JSON string
        return "'" + json.dumps(value).replace("'", "''") + "'"
    else:
        return "'" + str(value).replace("'", "''") + "'"

def display_and_download_results(test_data, original_format):
    """Display results and provide download options"""
    st.success("✅ Test Data Generated Successfully!")

    # JSON Preview
    with st.expander("📊 Preview Generated Data (JSON)"):
        st.json(test_data)

    sql_script = ""
    if "tables" in test_data:
        for table in test_data["tables"]:
            table_name = table.get("table_name", "unnamed_table")
            
            # Try to get data from "data" key; if not found, fall back to "columns"
            rows = table.get("data")
            
            # Heuristic: if "data" is missing but "columns" look like rows, use that
            if not rows and isinstance(table.get("columns"), list):
                first_col_item = table["columns"][0]
                if isinstance(first_col_item, dict) and all(not isinstance(v, dict) for v in first_col_item.values()):
                    rows = table["columns"]

            if not rows:
                print(f"⚠️ No usable rows found in table: {table_name}")
                continue

            print(f"✅ Generating SQL for table: {table_name}")
            print("🔍 Sample row:", rows[0])

            columns = rows[0].keys()
            col_list = ", ".join(f"`{col}`" for col in columns)

            for row in rows:
                values = ", ".join(format_sql_value(row.get(col)) for col in columns)
                sql_script += f"INSERT INTO `{table_name}` ({col_list}) VALUES ({values});\n"


    # SQL Preview
    with st.expander("📝 Preview SQL Script"):
        st.code(sql_script or "-- No SQL generated", language="sql")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(test_data, indent=2),
            file_name="test_data.json",
            mime="application/json"
        )

    with col2:
        st.download_button(
            label="⬇️ Download SQL Script",
            data=sql_script,
            file_name="test_data.sql",
            mime="text/sql"
        )