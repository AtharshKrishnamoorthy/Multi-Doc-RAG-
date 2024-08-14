# MultiDoc RAG System

## Demo Video

https://github.com/user-attachments/assets/28c49cb0-a99d-444c-95ab-7280b303c757

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

The MultiDoc RAG System provides the following features:

1. **Multi-Document Support**: Upload and process documents in various formats including Text, PDF, CSV, and Web URLs.
2. **Dynamic Document Loading**: Efficiently load and manage documents based on user input.
3. **Vector Store Creation**: Generate and save vector embeddings for documents using Hugging Face Embeddings and FAISS Vector store.
4. **Contextual Querying**: Use a pre-trained language model to answer queries based on the uploaded documents.
5. **Interactive Interface**: User-friendly interface built with Streamlit for seamless interaction.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Streamlit
- LangChain (including `langchain_community`, `langchain_huggingface`, `langchain_groq`, `langchain_core`)
- FAISS
- dotenv
- pygame (for text-to-speech, if needed)
- BeautifulSoup (bs4)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AtharshKrishnamoorthy/Multi-Doc-RAG
    cd Multi-Doc-RAG
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
   - Create a `.env` file in the project directory.
   - Add your Deepgram API key to the `.env` file:

     ```
     DEEPGRAM_API_KEY=your_deepgram_api_key
     ```

## Usage

To run the MultiDoc RAG System:

1. Navigate to the project directory.
2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` (or the address provided in the terminal).

### Document Upload

1. Use the sidebar to upload files or provide URLs for different types of documents (Text, Web, PDF, CSV).
2. The application will automatically handle document loading and processing.

### Querying

1. Enter your queries in the chat input box.
2. The system will provide responses based on the context of the uploaded documents.

## Model Training

The model used in this project is a pre-trained language model provided by Groq Inference Engine.It actually uses LPU (Language Processing Unit) to provide rapid inferences. The training specifics of the underlying model are managed externally and are not part of this repository.

### Model Details:

- **Model Name**: `llama-3.1-70b-versatile`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

## Configuration

Ensure you have the required environment setup for running the app. Modify `app.py` as needed for additional configurations.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the original branch: `git push origin feature-branch-name`.
5. Create a pull request.

Please update tests as appropriate and adhere to the project's coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions, feel free to reach out:

- Project Maintainer: Atharsh K
- Email: atharshkrishnamoorthy@gmail.com
- Project Link: [GitHub Repository](https://github.com/AtharshKrishnamoorthy/Multi-Doc-RAG)
