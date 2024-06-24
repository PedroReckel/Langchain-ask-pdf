from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import boto3
from io import BytesIO

app = Flask(__name__)
load_dotenv()

# Carregar variáveis de ambiente
FILE_SERVER_URL = os.getenv('FILE_SERVER_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Configurar boto3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def list_pdfs_from_s3(bucket_name):
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    pdf_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]
    return pdf_files

def download_pdf_from_s3(bucket_name, key):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    pdf_content = response['Body'].read()
    return pdf_content

def extract_text_from_pdf(pdf_content):
    pdf_reader = PdfReader(BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Endpoint para processar todos os PDFs do S3 e fazer uma pergunta
@app.route('/ask-pdf', methods=['POST'])
def ask_pdf():
    try:
        data = request.get_json()
        user_question = data['question']

        # Listar PDFs disponíveis no S3
        pdf_files = list_pdfs_from_s3(S3_BUCKET_NAME)
        
        all_text = ""
        
        for pdf_file in pdf_files:
            # Baixar cada PDF do S3
            pdf_content = download_pdf_from_s3(S3_BUCKET_NAME, pdf_file)
            
            # Extrair o texto do PDF
            text = extract_text_from_pdf(pdf_content)
            all_text += text + "\n"
        
        # Dividir o texto em chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)
        
        # Criar embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Adicionar instrução personalizada
        instruction = "Você é o Atendimentoel, uma IA especializada nos serviços da empresa EL Produções de Software. Com base nas perguntas que forem feitas, você irá responder com base nos PDFs disponíveis."
        prompt = f"{instruction}\n\n{user_question}"
        
        # Fazer a pergunta
        docs = knowledge_base.similarity_search(prompt)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=prompt)
            print(cb)
        
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
