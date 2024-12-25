# DocuQBot: AI Conversational Assistant for Document Analysis

This project is a PDF-based chatbot that allows users to upload PDF documents, extract text, and interact with the content through a conversational interface. Using FAISS for vector search and a lightweight Hugging Face model (like GPT-Neo-125M), the chatbot answers user queries based on the uploaded document. The system processes text into chunks, embeds them, and retrieves relevant sections for context-aware responses.

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**: Run `git clone https://github.com/nihilisticneuralnet/DocuQBot.git` to clone the repository to your local machine.

2. **Install Dependencies**: Navigate to the project directory and install the required packages by running `cd <repository-directory>` followed by `pip install -r requirements.txt`. 

3. **Set Up Environment Variables**: In the `.env` file in the project root directory and insert your Hugging Face API Token follows:
   ```plaintext
   HF_API_KEY=<your_hf_api>
   ```
   Replace `<your_hf_api>` with your actual API keys.

4. **Run the Application**: Finally, run the application using Streamlit by executing `python -m streamlit run app.py`.

Ensure you have all the necessary libraries installed before running these commands.
