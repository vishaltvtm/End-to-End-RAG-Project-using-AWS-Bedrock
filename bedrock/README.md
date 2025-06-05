# tools
1. Langchain
2. PyPDF
3. streamlit
4. FAISS
5. Bedrock
6. LLama 2
7. boto3

# How to run
1. Install the required packages:
   ```bash
   uv sync
   ```
2. Run the Streamlit app:
   ```bash
    streamlit run app.py
    ```
# How to use
1. Upload a PDF document using the provided interface.  
2. The app will process the document and extract text using PyPDF.
3. The extracted text will be indexed using FAISS for efficient retrieval.
4. You can then query the indexed text using the Bedrock model.
5. The app will display the results of your query, leveraging the capabilities of the LLama 2 model for natural language understanding and response generation.
# Requirements
- Python 3.8 or higher
- Streamlit
- PyPDF2
- FAISS
- Bedrock SDK
# Example Usage
- Upload a PDF document containing information about a specific topic.
- Enter a query related to the content of the PDF.
- The app will return relevant information extracted from the PDF, processed through the LLama 2 model.
# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
# Contact
For any questions or feedback, please contact the project maintainer at [
    vinayak tavatam ]
