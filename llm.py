import os
from typing import List, Any, Optional, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from bigquery_vector import create_vectors as bq_create_vectors, search_by_text as bq_search_by_text

PROJECT_ID = "GCP_PROJECT_ID"
REGION = "REGION"
DATASET = "BigQuery_Datset_Name"
TABLE = "BigQuery_Table_Name"

service_account_path = r"PATH_TO_SERVICE_ACCOUNT"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

LLM = ChatOpenAI()


class BigQueryRetriever(BaseRetriever):
    project_id: str
    region: str
    dataset: str
    table: str
    filter: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_id = kwargs['project_id']
        self.region = kwargs['region']
        self.dataset = kwargs['dataset']
        self.table = kwargs['table']
        self.filter = kwargs['filter']

    def get_relevant_documents(self, query: str) -> List[Document]:
        return bq_search_by_text(
            project_id=self.project_id,
            region=self.region,
            dataset=self.dataset,
            table=self.table,
            filter=self.filter,
            query=query
        )

# Will be required if data needs to be scraped from Internet (Whirlpool's Website/Site Pages)

# def ingest_vectors(url, context, truncate_all=False):
#     loader = WebBaseLoader(url)
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter()
#     documents = text_splitter.split_documents(docs)
#     metadata = [{"length": len(d.page_content), "context": context} for d in documents]

#     bq_create_vectors(
#         project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE,
#         metadata=metadata, texts=[d.page_content for d in documents], truncate=truncate_all
#     )

# For only PDFs

# def ingest_vectors(pdf_path, context, truncate_all=False):
#     # Check if the PDF file exists
#     if not os.path.exists(pdf_path):
#         raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")
    
#     # Load documents from the PDF file
#     loader = PyMuPDFLoader(pdf_path)
#     docs = loader.load()

#     # Split loaded documents into chunks if necessary
#     text_splitter = RecursiveCharacterTextSplitter()
#     documents = text_splitter.split_documents(docs)
    
#     # Generate metadata for each document chunk
#     metadata = [{"length": len(d.page_content), "context": context} for d in documents]

#     # Create vectors and upload to BigQuery
#     bq_create_vectors(
#         project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE,
#         metadata=metadata, texts=[d.page_content for d in documents], truncate=truncate_all
#     )

def ingest_vectors(file_path, context, truncate_all=False):
    # Determine the file type based on the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Initialize the loader based on file type
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Load documents from the file
    docs = loader.load()

    # Split loaded documents into chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    # Generate metadata for each document chunk
    metadata = [{"length": len(d.page_content), "context": context} for d in documents]

    # Create vectors and upload to BigQuery
    bq_create_vectors(
        project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE,
        metadata=metadata, texts=[d.page_content for d in documents], truncate=truncate_all
    )
    
    

def search(query, filter):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(LLM, prompt)
    retriever = BigQueryRetriever(project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE, filter=filter)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

# Run it only when new data needs to be ingested

# ingest_vectors(file_path= r"C:\Users\NAIRTS\Downloads\Financials Sample Data.csv", context= "test")

# Context is just an identifier for the embeddings in BQ Table (Check if we need it)
answer = search(query=input("Enter your query: "), filter={"context": "test"})
print(answer)