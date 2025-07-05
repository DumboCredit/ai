# imports
from fastapi import FastAPI, Request
import logging
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn 
import os

# env vars
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# langchain
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="credit_collection",
    embedding_function=embeddings,
    persist_directory="./credit_db",  # Where to save data locally, remove if not necessary
    collection_metadata={"hnsw:space": "cosine"}
)

# api
app = FastAPI(
    title= "DumboAI",
    version="0.1"
)

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "path": request.url.path
        }
    )

# models
# borrower
class BORROWER(BaseModel):
    FirstName: str
    LastName: str
    BirthDate: str

# credit score
class FACTOR(BaseModel):
    Code: str
    Text: str
class POSITIVE_FACTOR(BaseModel):
    Code: str
    Text: str
class CREDIT_SCORE(BaseModel):
    Date: str
    Value: str
    CreditRepositorySourceType: str
    RiskBasedPricingMax: str
    RiskBasedPricingMin: str
    RiskBasedPricingPercent: str
    FACTOR: list[FACTOR]
    POSITIVE_FACTOR: list[POSITIVE_FACTOR]

# credit inquiry
class CREDIT_REPOSITORY(BaseModel):
    SourceType: str
class CREDIT_INQUIRY(BaseModel):
    Date: str
    Name: str
    RawIndustryText: str
    CreditInquiryID: str
    CREDIT_REPOSITORY: CREDIT_REPOSITORY

# credit summary
class DATA_SET_SUMMARY(BaseModel):
    ID: str
    Name: str
    Value:str
class CREDIT_SUMMARY(BaseModel):
    DATA_SET: list[DATA_SET_SUMMARY]

# request
class CreditRequest(BaseModel):
    USER_ID: str
    BORROWER: Optional[BORROWER]	
    CREDIT_SCORE:list[CREDIT_SCORE]
    CREDIT_INQUIRY: list[CREDIT_INQUIRY]
    CREDIT_SUMMARY_EFX: Optional[CREDIT_SUMMARY] = None #Equifax
    CREDIT_SUMMARY_TUI: Optional[CREDIT_SUMMARY] = None #TransUnion
    CREDIT_SUMMARY_XPN: Optional[CREDIT_SUMMARY] = None #Experian

# routes
@app.post("/add-user-credit-data")
async def add_user_credit_data(historic_credit:CreditRequest):
        documents = []
         # load personal data
        if not historic_credit.BORROWER.FirstName is None:
            documents.append(
                Document(
                    page_content= f"My firstName: {historic_credit.BORROWER.FirstName}",
                    metadata={"field": "FirstName", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="FirstName",
                )
            )
        if not historic_credit.BORROWER.LastName is None:
            documents.append(
                Document(
                    page_content= f"My lastName: {historic_credit.BORROWER.LastName}",
                    metadata={"field": "LastName", "source": "Personal Info", "user_id": historic_credit.USER_ID},
                    id='LastName',
                )
            )
        # load credit inquiry
        if not historic_credit.CREDIT_INQUIRY is None:
            for i in historic_credit.CREDIT_INQUIRY:
                documents.append(Document(
                    page_content= f"Inquiry: {i.Name}; From credit car: {i.CREDIT_REPOSITORY.SourceType}; Date: {i.Date}",
                    metadata={"source": "CreditInquiry", "credit_card": i.CREDIT_REPOSITORY.SourceType, "date": i.Date, "user_id": historic_credit.USER_ID},
                    id=i.CreditInquiryID,
                ))
        # load credit summary
        if not historic_credit.CREDIT_SUMMARY_EFX is None:
            for i in historic_credit.CREDIT_SUMMARY_EFX.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From credit Card: Equifax",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_TUI is None:
            for i in historic_credit.CREDIT_SUMMARY_TUI.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From credit Card: TransUnion",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_XPN is None:
            for i in historic_credit.CREDIT_SUMMARY_XPN.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From credit Card: Experian",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))

        # load credit score
        if not historic_credit.CREDIT_SCORE is None:
            for i in historic_credit.CREDIT_SCORE:
                credit_card = i.CreditRepositorySourceType
                date_of_credit_score = i.Date
                for j in i.FACTOR:
                    documents.append(Document(
                        page_content= f"Credit Score Factor ({date_of_credit_score}):  {j.Text}. From Credit Card: {credit_card}",
                        metadata={"source": "CreditScore", "field": "factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID },
                        id=j.Code,
                    ))
                for j in i.POSITIVE_FACTOR:
                    documents.append(Document(
                        page_content= f"Credit Score Positive Factor ({date_of_credit_score}):  {j.Text}. From Credit Card: {credit_card}",
                        metadata={"source": "CreditScore", "field": "positive_factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID },
                        id=j.Code,
                    ))
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
  
# model
class QueryRequest(BaseModel):
    user_id: str
    query: str
# endpoint to retrieve an answer
@app.post("/query")
async def query(query_request:QueryRequest):
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2, "k": 20, "filter": {"user_id": query_request.user_id} }
    )
    llm = ChatOpenAI()
    system_prompt = (
        f"Work as an assistant about credit to myself"
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "keep the answer concise. "
        "Only respond with the data and some concept if it necesary, dont give advice."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain.invoke({"input": query_request.query}).answer

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8080)