# GenAI Test Case Generator

A powerful Streamlit-based application that leverages Generative AI to automatically generate comprehensive test cases from user stories and business requirements documents (BRDs). This tool helps QA engineers and developers save time by automating the test case creation process while ensuring comprehensive test coverage.

## Features

- **AI-Powered Test Generation**: Uses advanced language models (OpenAI GPT-5.1 or local Llama3) to generate detailed test cases
- **Multi-Format Support**: Processes Excel files containing user stories and features
- **Comprehensive Coverage**: Generates positive, negative, and edge case test scenarios
- **Multiple Modules**:
  - Test Case Generator: Core functionality for generating test cases from user stories
  - BRD Reader: Process and analyze Business Requirements Documents
  - Test Data Generator: Generate test data for various scenarios
  - Automation Script: Create automated test scripts
  - Coverage Validator: Validate test coverage completeness
  - Mock API Generator: Generate mock APIs for testing
- **Export Capabilities**: Save generated test cases to Excel format
- **User-Friendly Interface**: Clean Streamlit web interface with navigation
- **Docker Support**: Easy containerized deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-5.1 model) or local Ollama installation (for Llama3)
- Git

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/parody25/TestCaseGenerator.git
cd TestCaseGenerator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env` (if available) or create a new `.env` file
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/parody25/TestCaseGenerator.git
cd TestCaseGenerator
```

2. Build the Docker image:
```bash
docker build -t testcase-generator .
```

3. Run the container:
```bash
docker run -p 8050:8050 testcase-generator
```

The application will be available at `http://localhost:8050`

## Usage

1. **Upload Excel File**: Use the file uploader to upload an Excel file containing user stories and features
2. **Select Sheet**: Choose the appropriate sheet from your Excel file
3. **Choose AI Model**: Select between "gpt-5.1" (requires OpenAI API key) or "llama3" (requires local Ollama)
4. **Add Context**: Provide additional context in the text area if needed
5. **Generate Test Cases**: Click "Generate Test Cases" to create comprehensive test scenarios
6. **Export Results**: Use "Save to Excel" to download the generated test cases

### Navigation

The application provides a sidebar navigation for different modules:
- **Welcome**: Landing page with overview
- **Test Case Generator**: Main test case generation functionality
- **BRD Reader**: Business requirements document processing
- **Test Data Generator**: Generate test data
- **Automation Script**: Create automated test scripts
- **Coverage Validator**: Validate test coverage
- **Mock API Generator**: Generate mock APIs

## Requirements

Key dependencies include:
- `streamlit>=1.35.0` - Web application framework
- `openai>=1.25.1` - OpenAI API client
- `llama-index>=0.10.14` - Document processing and indexing
- `langchain>=0.1.17` - LLM framework
- `pandas>=2.2.2` - Data manipulation
- `faiss-cpu>=1.8.0` - Vector similarity search
- `pymupdf>=1.23.21` - PDF processing

See `requirements.txt` for the complete list of dependencies.

## Project Structure

```
TestCaseGenerator/
├── app.py                      # Main Streamlit application
├── test_case_generator2.py     # Core test case generation logic
├── brd_reader.py              # BRD processing functionality
├── test_data_generator.py     # Test data generation
├── automation_script.py       # Automation script creation
├── test_coverage.py           # Coverage validation
├── mock_api_generator.py      # Mock API generation
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # This file
├── sample_user_stories.json   # Sample data
├── test_case_schema.json      # Test case schema
└── embeddings/               # Vector embeddings storage
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project does not currently specify a license. Please check with the repository owner for usage permissions.

## Support

For issues, questions, or contributions, please create an issue in the GitHub repository.
