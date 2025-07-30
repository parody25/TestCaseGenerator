import streamlit as st
import json
import yaml
from openai import OpenAI
from io import StringIO
from dotenv import load_dotenv
import os
import random
import string
import uuid
from datetime import datetime, timedelta

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def mock_api_generator():
    st.title("🧪 Enhanced Mock API Generator")
    st.markdown("""
    **Generate fully functional mock APIs** with realistic responses based on:
    - Uploaded OpenAPI specs
    - Natural language descriptions
    """)

    mode = st.radio("Choose input method:", ["Upload OpenAPI Spec", "Describe API in Natural Language"], horizontal=True)

    if mode == "Upload OpenAPI Spec":
        handle_file_upload()
    else:
        handle_natural_language_input()

def handle_file_upload():
    uploaded_file = st.file_uploader("Upload OpenAPI spec (.yaml or .json)", type=["yaml", "yml", "json"])
    if uploaded_file:
        try:
            content = uploaded_file.read().decode("utf-8")
            if uploaded_file.name.endswith(".json"):
                spec = json.loads(content)
            else:
                spec = yaml.safe_load(content)

            st.success("✅ Specification parsed successfully")

            with st.expander("View Parsed OpenAPI Spec"):
                st.json(spec)

            if st.button("🔧 Generate Enhanced Mock Server", type="primary"):
                with st.spinner("Generating realistic mock server..."):
                    code = generate_enhanced_mock_from_spec(spec)

                    st.subheader("🚀 Complete Mock Server Implementation")
                    st.code(code, language="python")

                    st.download_button(
                        label="⬇️ Download Mock Server",
                        data=code,
                        file_name="mock_api.py",
                        mime="text/x-python"
                    )

        except Exception as e:
            st.error(f"❌ Error parsing spec: {str(e)}")

def handle_natural_language_input():
    desc = st.text_area("Describe your API (e.g., 'A user service with login, registration, and profile endpoints')",
                        height=150)

    if st.button("🧠 Generate Complete API Implementation", type="primary"):
        if not desc.strip():
            st.warning("Please enter an API description")
            return

        with st.spinner("Generating OpenAPI spec and mock implementation..."):
            try:
                spec = generate_openapi_from_description(desc)
                code = generate_enhanced_mock_from_spec(spec)

                st.subheader("📦 Generated OpenAPI Specification")
                with st.expander("View OpenAPI Spec"):
                    st.json(spec)

                st.subheader("🚀 Complete Mock Server Implementation")
                st.code(code, language="python")

                st.download_button(
                    label="⬇️ Download Mock Server",
                    data=code,
                    file_name="mock_api.py",
                    mime="text/x-python"
                )

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")

def generate_openapi_from_description(desc):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an OpenAPI specification generator. 
             Return ONLY the raw OpenAPI 3.0 JSON specification based on the description.
             Include detailed schemas, parameters, and response examples.
             Format: Pure JSON with no markdown or additional text."""},
            {"role": "user", "content": desc}
        ],
        temperature=0.3
    )

    raw_json = response.choices[0].message.content.strip()

    # Clean JSON output
    if raw_json.startswith("```json"):
        raw_json = raw_json[7:-3].strip()
    elif raw_json.startswith("```"):
        raw_json = raw_json[3:-3].strip()

    return json.loads(raw_json)

def generate_enhanced_mock_from_spec(spec):
    code = [
        "from fastapi import FastAPI, HTTPException, Depends, Header, status",
        "from pydantic import BaseModel",
        "from typing import Optional, List, Dict",
        "import uuid",
        "import random",
        "from datetime import datetime, timedelta\n",
        "app = FastAPI(title=\"Mock API Server\", description=\"Auto-generated mock API\")\n",
        "# --- Data Models ---"
    ]

    components = spec.get("components", {}).get("schemas", {})
    for model_name, schema in components.items():
        code.append(f"\nclass {model_name}(BaseModel):")
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                py_type = get_python_type(prop_schema.get("type", "string"))
                code.append(f"    {prop}: {py_type}")
        else:
            code.append("    pass")

    code.extend([
        "\n# --- Mock Database ---",
        "mock_db = {}",
        "def initialize_mock_data():",
        "    pass\n"
    ])

    code.append("\n# --- API Endpoints ---")
    paths = spec.get("paths", {})
    for path, methods in paths.items():
        for method, details in methods.items():
            operation_id = details.get("operationId", f"{method}_{path[1:].replace('/', '_')}")
            summary = details.get("summary", f"Mock {method.upper()} {path}")
            code.append(f"\n@app.{method}(\"{path}\")")
            code.append(f"async def {operation_id}():")
            code.append(f"    \"\"\"{summary}\"\"\"")

            response_schema = get_response_schema(details)
            mock_response = generate_mock_response(response_schema)
            code.append(f"    return {json.dumps(mock_response, indent=4)}")

    code.extend([
        "\n# Initialize mock data",
        "initialize_mock_data()",
        "\nif __name__ == \"__main__\":",
        "    import uvicorn",
        "    uvicorn.run(app, host=\"0.0.0.0\", port=8000)"
    ])

    return "\n".join(code)

def get_python_type(json_type):
    type_map = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "List",
        "object": "Dict"
    }
    return type_map.get(json_type, "Any")

def get_response_schema(endpoint_spec):
    responses = endpoint_spec.get("responses", {})
    if "200" in responses:
        content = responses["200"].get("content", {})
        if "application/json" in content:
            return content["application/json"].get("schema", {})
    return {}

def generate_mock_response(schema):
    if not schema:
        return {"message": "Success", "data": {}}

    if "example" in schema:
        return schema["example"]

    mock_data = {}
    if "properties" in schema:
        for prop, prop_schema in schema["properties"].items():
            mock_data[prop] = generate_mock_value(prop_schema)

    return mock_data

def generate_mock_value(prop_schema):
    prop_type = prop_schema.get("type", "string")
    format = prop_schema.get("format", "")

    if prop_type == "string":
        if format == "date-time":
            return (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif format == "email":
            return f"user{random.randint(1, 100)}@example.com"
        elif format == "uuid":
            return str(uuid.uuid4())
        else:
            return "".join(random.choices(string.ascii_letters + string.digits, k=10))
    elif prop_type == "integer":
        return random.randint(1, 100)
    elif prop_type == "number":
        return round(random.uniform(1, 100), 2)
    elif prop_type == "boolean":
        return random.choice([True, False])
    elif prop_type == "array":
        item_schema = prop_schema.get("items", {"type": "string"})
        return [generate_mock_value(item_schema) for _ in range(random.randint(1, 5))]
    elif prop_type == "object":
        return generate_mock_response(prop_schema)
    return None
