# PDF-Chatbot
---
## Prerequisites for using the app
- API key from Gemini.
---
## Libraries Required

1. **streamlit** - used for UI development 
2. **PyPDF2** [PdfReader] - used to read the PDF 
3. **langchain**
   
	a. **text_splitter** [RecursiveCharacterTextSplitter] - used for Chunking the data read from PDF

	b. **google_genai**

		i) GoogleGenerativeAIEmbeddings - used for creating the embedding of each chunk
		ii) ChatGoogleGenerativeAI - chatbot functionality

	c. **vectorstores** [FAISS] - used for storing the embedding in vectordb

	d. **chains.question_answering** [load_qa_chain] - chat history

	e. **prompts** [PromptTemplate] - sets the prompt features


5. **dotenv** [load_dotenv] - loads the env variables
---
## Steps

1. load the env variable and pass the "GOOGLE_API_KEY" to the LLM 
2. text extraction from pdf 
3. chunk the text 
4. create embedding and store in vectordb
5. give prompt and define the model
6. take user question and similarity search with FAISSdb 
7. get response
---

