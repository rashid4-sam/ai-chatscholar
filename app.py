# Import necessary libraries
import os, re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


start_greeting = ["hi","hello"]
end_greeting = ["bye"]
way_greeting = ["who are you?"]

#Using this folder for storing the uploaded docs. Creates the folder at runtime if not present
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

# OpenRouter client for chat
openai_client = ChatOpenAI(
    api_key = "api_key",
    model = "meta-llama/llama-3.3-8b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
)

class HumanMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'HumanMessage(content={self.content})'

class AIMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'AIMessage(content={self.content})'


def get_pdf_text(pdf_docs):
    text = ""
    pdf_txt = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR,pdf.filename)
        pdf_txt = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            pdf_txt += page.extract_text()

        with (open(filename, "w", encoding="utf-8")) as op_file:
            op_file.write(pdf_txt)

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Use HuggingFace embeddings (free, no API key needed)
    # Using paraphrase-MiniLM-L3-v2 - smaller and faster model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={'device': 'cpu'}  # Use CPU to avoid GPU issues
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Use the OpenRouter ChatOpenAI for conversation
    llm = ChatOpenAI(
        api_key="api_key",
        model="google/gemini-2.5-flash",
        openai_api_base="https://openrouter.ai/api/v1",
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def _grade_essay(essay):
    global rubric_text

    # Construct the full prompt
    system_prompt = (
        "You are an expert essay grader. Carefully grade the essay based on the rubric below "
        "and provide a detailed evaluation.\n\nRUBRIC:\n"
        f"{rubric_text}\n\n"
    )
    user_prompt = "ESSAY:\n" + essay

    full_prompt = system_prompt + user_prompt

    response = openai_client.invoke(full_prompt)
    data = response.content if hasattr(response, "content") else str(response)
    data = re.sub(r'\n', '<br>', data)

    return data



@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    try:
        pdf_docs = request.files.getlist('pdf_docs')
        
        # Check if files were uploaded
        if not pdf_docs or pdf_docs[0].filename == '':
            return "No files uploaded", 400
        
        print("Extracting text from PDFs...")
        raw_text = get_pdf_text(pdf_docs)
        
        print("Creating text chunks...")
        text_chunks = get_text_chunks(raw_text)
        
        print("Creating vector store (this may take a moment on first run)...")
        vectorstore = get_vectorstore(text_chunks)
        
        print("Setting up conversation chain...")
        conversation_chain = get_conversation_chain(vectorstore)
        
        print("Processing complete!")
        return redirect('/chat')
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history
    msgs = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        
    return render_template('new_chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    text = ""
    if request.method == 'POST':
        if request.form.get('essay_rubric', False):
            global rubric_text
            rubric_text = request.form.get('essay_rubric')

            return render_template('new_essay_grading.html')
        
        if len(request.files['file'].filename) > 0:
            pdf_file = request.files['file']
            text = extract_text_from_pdf(pdf_file)
            result = _grade_essay(text)
        else:
            text = request.form.get('essay_text')
            result = _grade_essay(text)
    
    return render_template('new_essay_grading.html', result=result, input_text=text)
    
@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    # Use debug=False or use_reloader=False to prevent constant reloading
    app.run(debug=True, use_reloader=False)
