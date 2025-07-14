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
import datetime

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
        - MUST maintain this exact structure:
        {{
            "schema": <original_schema_definition>,
            "data": [
            {{...record1...}},
            {{...record2...}}
            ],
            "_metadata": {{...}}
        }}
        - MUST include the original schema definition in the 'schema' field
        - MUST include generated data in the 'data' field as an array of objects
        - Return ONLY the JSON object (no markdown, no text)
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a sophisticated test data generation system."},
            {"role": "user", "content": prompt}
        ],
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

def format_sql_value(value):
    """Safely format values for SQL insert with more type awareness"""
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        # Handle date/time strings that don't need quoting
        if re.match(r'^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$', value):
            return f"'{value}'"
        return "'" + value.replace("'", "''") + "'"
    elif isinstance(value, (bool)):
        return str(value).upper()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (dict, list)):
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return "'" + json_str.replace("'", "''") + "'"
        except:
            return "'" + str(value).replace("'", "''") + "'"
    elif isinstance(value, (datetime.date, datetime.datetime)):
        return f"'{value.isoformat()}'"
    else:
        return "'" + str(value).replace("'", "''") + "'"

def generate_sql_from_test_data(test_data):
    """Generate SQL script from test data with flexible structure handling"""
    sql_script = ""
    
    if not isinstance(test_data, dict):
        return "-- Unsupported test data format"
    
    # Handle the new structure with separate schema and data
    if "schema" in test_data and "data" in test_data:
        table_name = test_data["schema"].get("table_name", "unnamed_table")
        columns_def = test_data["schema"].get("columns", [])
        rows = test_data["data"]
        
        # Generate CREATE TABLE from schema
        sql_script += generate_create_table(table_name, columns_def)
        
        # Generate INSERT statements from data
        columns = [col["name"] for col in columns_def] if isinstance(columns_def, list) else list(rows[0].keys())
        for row in rows:
            sql_script += generate_row_as_sql(row, table_name, columns)
    
    # Handle the standard case (maintain backward compatibility)
    elif "tables" in test_data and isinstance(test_data["tables"], dict):
        for table_name, table_data in test_data["tables"].items():
            if isinstance(table_data, dict):
                sql_script += generate_table_sql({**table_data, "table_name": table_name})
            else:
                sql_script += generate_table_sql({
                    "table_name": table_name,
                    "data": table_data
                })
    
    # Handle the case where data is at the root level with columns definition
    elif all(key in test_data for key in ["table_name", "columns", "tables"]):
        columns = [col["name"] for col in test_data["columns"]]
        for row in test_data["tables"]:
            sql_script += generate_row_as_sql(row, test_data["table_name"], columns)
    
    else:
        sql_script += "-- Could not determine data structure\n"
        sql_script += f"-- Data keys: {list(test_data.keys())}\n"
    
    return sql_script

def generate_row_as_sql(row, table_name, columns=None):
    """Generate SQL for a single row"""
    if not isinstance(row, dict):
        return f"-- Invalid row data for table {table_name}\n"
    
    if columns is None:
        columns = list(row.keys())
    
    available_cols = [col for col in columns if col in row]
    values = [format_sql_value(row.get(col)) for col in available_cols]
    col_list = ", ".join(f'"{col}"' for col in available_cols)
    value_list = ", ".join(values)
    return f'INSERT INTO "{table_name}" ({col_list}) VALUES ({value_list});\n'

def generate_table_sql(table):
    """Generate SQL for a single table with enhanced structure detection"""
    table_name = table.get("table_name", "unnamed_table")
    table_sql = f"-- Table: {table_name}\n"
    
    # Try multiple ways to find rows
    rows = []
    
    # Case 1: Direct 'data' key
    if "data" in table and isinstance(table["data"], list):
        rows = table["data"]
    
    # Case 2: Columns format where each column is a list of values
    elif "columns" in table and isinstance(table["columns"], dict):
        try:
            columns = table["columns"]
            row_count = min(len(v) for v in columns.values()) if columns else 0
            rows = [
                {col: columns[col][i] for col in columns}
                for i in range(row_count)
            ]
        except Exception as e:
            table_sql += f"-- Failed to convert columns: {str(e)}\n"
    
    # Case 3: Root level data (for backward compatibility)
    elif all(isinstance(v, (list, dict)) for v in table.values()):
        rows = [dict(zip(table.keys(), values)) for values in zip(*table.values())]
    
    if not rows:
        table_sql += f"-- No usable rows found in table {table_name}\n"
        table_sql += f"-- Table keys: {list(table.keys())}\n"
        return table_sql
    
    # Generate column list from first row
    first_row = rows[0]
    columns = list(first_row.keys())
    
    # Add CREATE TABLE if schema is available
    if "columns" in table and isinstance(table["columns"], list):
        table_sql += generate_create_table(table_name, table["columns"])
    
    # Generate INSERT statements
    for row in rows:
        values = [format_sql_value(row.get(col)) for col in columns]
        col_list = ", ".join(f'"{col}"' for col in columns)
        value_list = ", ".join(values)
        table_sql += f'INSERT INTO "{table_name}" ({col_list}) VALUES ({value_list});\n'
    
    return table_sql

def generate_create_table(table_name, columns_def):
    """Generate CREATE TABLE statement from columns definition"""
    if not columns_def or not isinstance(columns_def, list):
        return f"-- No valid column definitions found for table {table_name}\n"
    
    create_sql = f"CREATE TABLE \"{table_name}\" (\n"
    columns = []
    
    for col_def in columns_def:
        if not isinstance(col_def, dict):
            continue
            
        col_name = col_def.get("name", "unnamed_column")
        col_type = col_def.get("type", "text").lower()
        
        # Map generic types to SQL types
        type_mapping = {
            "string": "VARCHAR(255)",
            "integer": "INTEGER",
            "float": "NUMERIC(15,2)",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "timestamp": "TIMESTAMP",
            "text": "TEXT"
        }
        sql_type = type_mapping.get(col_type, "TEXT")
        
        # Add constraints
        constraints = []
        if col_def.get("constraints", {}).get("unique"):
            constraints.append("UNIQUE")
        if col_def.get("constraints", {}).get("not_null"):
            constraints.append("NOT NULL")
        if "enum" in col_def:
            enum_values = ",".join(f"'{v}'" for v in col_def["enum"])
            constraints.append(f"CHECK ({col_name} IN ({enum_values}))")
        if "min" in col_def.get("constraints", {}):
            constraints.append(f"CHECK ({col_name} >= {col_def['constraints']['min']})")
        if "max" in col_def.get("constraints", {}):
            constraints.append(f"CHECK ({col_name} <= {col_def['constraints']['max']})")
        
        columns.append(f'    "{col_name}" {sql_type} {" ".join(constraints)}'.strip())
    
    create_sql += ",\n".join(columns)
    create_sql += "\n);\n\n"
    return create_sql

def display_and_download_results(test_data, original_format):
    """Display results and provide download options"""
    st.success("✅ Test Data Generated Successfully!")

    # JSON Preview
    with st.expander("📊 Preview Generated Data (JSON)"):
        st.json(test_data)

    try:
        sql_script = generate_sql_from_test_data(test_data)
        
        # SQL Preview with debug info
        with st.expander("📝 Preview SQL Script"):
            if "-- No usable rows found in any format" in sql_script:
                st.warning("No data was found for some tables. Check the debug information below.")
            st.code(sql_script, language="sql")

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
            
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        st.error(f"Problem occurred with data structure: {str(test_data)[:500]}...")
        st.stop()