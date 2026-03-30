# Project List

## Chapter 1

- `ch_1projects.ipynb`  
  A notebook with a chatbot that uses a chat template to answer questions about pasted context.

- `chatbot_app.py`  
  A simple baseline chatbot application with only a name defined.

## Chapter 2 & 3
- `rag_cli.py`  
  This notebook demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline. We generate five example documents with an AI model, chunk them, and store their embeddings in a PostgreSQL database extended with pgvector (running in Docker). Using LangChain, Ollama, and PGVector, we then build a chatbot that answers questions grounded in those stored documents.

- `rag_cli.py`  
  This script builds a simple command-line chatbot on top of an existing PGVector database. It connects to the stored document embeddings, retrieves the most relevant chunks for each user question, and then uses an Ollama language model to generate a short answer based only on the retrieved context.The chatbot runs in a loop, so you can keep asking questions until you type `exit` or `quit`. This makes it a minimal but practical example of how a Retrieval-Augmented Generation (RAG) system can be turned into an interactive local application.
## Chapter 4

- `main.ipynb`  
  A notebook that loads colleague PDF profiles, splits them into chunks, creates embeddings, and stores them in a pgvector PostgreSQL database.
- `rag_core.py`  
  The core RAG pipeline that connects PGVector retrieval, HuggingFace embeddings, LangGraph memory, and an Ollama model to answer questions with context.
- `docker-compose.yml`  
  A Docker Compose file that starts the pgvector PostgreSQL database used for this chapter.
- `rag_app.py`  
  A Flask web app that uses the RAG pipeline to provide a chatbot with retrieval and conversation memory.
### How to Run

1. Open a terminal and start the pgvector database:

   ```bash
   docker compose up
   ```

2. Open a second terminal and run the RAG chatbot app:

   ```bash
   python3 rag_app.py
   ```

3. Once the app starts, open the following link in your browser:

   [http://127.0.0.1:5000](http://127.0.0.1:5000)  

4. You should now see the chatbot interface and can start asking questions.

## Chapter 5

- `main.ipynb`  
  A notebook that builds **Rosenxt_bot**, a simple Telegram-integrated LLM project. It uses LangChain with Ollama to generate answers using `llama3.1:latest`, then sends both the user’s question and the model’s response to a Telegram supergroup or channel through the Telegram Bot API.