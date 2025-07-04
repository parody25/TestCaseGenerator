import streamlit as st
import openai
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def automation_script():
    st.title("⚙️ Automation Script Converter")
    st.markdown("Convert finalized user stories or test steps into automation scripts using GenAI.")

    user_input = st.text_area("📋 Paste User Story or Test Steps", height=250)

    framework = st.selectbox("💻 Select Automation Framework", ["Selenium (Python)", "TestNG (Java)", "Cypress (JavaScript)"])

    if st.button("🚀 Generate Script"):
        if not user_input.strip():
            st.warning("Please provide user story or test steps.")
            return

        prompt = build_prompt(user_input, framework)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a test automation expert who generates executable code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            script = response.choices[0].message.content
            st.subheader("🧾 Generated Script")
            st.code(script, language=get_language_from_framework(framework))

            # Enable file download
            file_extension = get_extension(framework)
            file_name = f"automation_script{file_extension}"
            st.download_button(
                label=f"⬇️ Download {file_extension.upper()} Script",
                data=script.encode("utf-8"),
                file_name=file_name,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Failed to generate script: {e}")

def build_prompt(steps: str, framework: str) -> str:
    if "Selenium" in framework:
        language = "Python"
        prompt = f"""
Convert the following user story or test steps into a Selenium script using {language}. Use appropriate selectors and add comments for each step.

Test Steps:
{steps}

Return only the code, no explanations.
"""
    elif "TestNG" in framework:
        language = "Java"
        prompt = f"""
Write a complete TestNG test case in {language} that automates the following test steps. Include imports, setup, and meaningful assertions.

Test Steps:
{steps}

Return only the code.
"""
    elif "Cypress" in framework:
        language = "JavaScript"
        prompt = f"""
Convert the following test steps into a Cypress test script in {language}. Use best practices like 'cy.get' and include validation checks.

Test Steps:
{steps}

Only return the Cypress code.
"""
    return prompt

def get_language_from_framework(framework: str) -> str:
    if "Selenium" in framework:
        return "python"
    elif "TestNG" in framework:
        return "java"
    elif "Cypress" in framework:
        return "javascript"
    return "text"

def get_extension(framework: str) -> str:
    if "Selenium" in framework:
        return ".py"
    elif "TestNG" in framework:
        return ".java"
    elif "Cypress" in framework:
        return ".js"
    return ".txt"
