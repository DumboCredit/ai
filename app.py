# imports
from fastapi import FastAPI, Request, HTTPException
import logging
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn 
import os

# env vars
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
class _BORROWER(BaseModel):
    FirstName: str

# credit score
class _FACTOR(BaseModel):
    Code: str
    Text: str
class _POSITIVE_FACTOR(BaseModel):
    Code: str
    Text: str
class _CREDIT_SCORE(BaseModel):
    Date: str
    Value: str
    CreditRepositorySourceType: str
    RiskBasedPricingMax: str
    RiskBasedPricingMin: str
    RiskBasedPricingPercent: str
    FACTOR: Optional[list[_FACTOR]] = None
    POSITIVE_FACTOR: Optional[list[_POSITIVE_FACTOR]] = None

# credit inquiry
class CREDIT_REPOSITORY(BaseModel):
    SourceType: str
class _CREDIT_INQUIRY(BaseModel):
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

# credit liability
class _CREDITOR(BaseModel):
    Name: str

class _PAYMENT_PATTERN(BaseModel):
    StartDate: str

class _HIGHEST_ADVERSE_RATING(BaseModel):
    Type: str

class _CREDIT_LIABILITY(BaseModel):
    CreditLiabilityID: str
    OriginalBalanceAmount: Optional[str] = None
    UnpaidBalanceAmount: Optional[str] = None
    MonthlyPaymentAmount: Optional[str] = None
    TermsMonthsCount: Optional[str] = None
    MonthsReviewedCount: Optional[str] = None
    CreditLoanType: Optional[str] = None
    CREDITOR: _CREDITOR 
    PAYMENT_PATTERN: _PAYMENT_PATTERN
    TermsDescription: Optional[str] = None
    CREDIT_REPOSITORY: Union[CREDIT_REPOSITORY, list[CREDIT_REPOSITORY]] 
    HIGHEST_ADVERSE_RATING: Optional[_HIGHEST_ADVERSE_RATING] = None

# request
class CreditRequest(BaseModel):
    USER_ID: str
    API_KEY: str
    BORROWER: Optional[_BORROWER] = None
    CREDIT_SCORE: Optional[list[_CREDIT_SCORE]] = None
    CREDIT_INQUIRY: Optional[list[_CREDIT_INQUIRY]] = None
    CREDIT_LIABILITY: Optional[list[_CREDIT_LIABILITY]] = None
    CREDIT_SUMMARY_EFX: Optional[CREDIT_SUMMARY] = None #Equifax
    CREDIT_SUMMARY_TUI: Optional[CREDIT_SUMMARY] = None #TransUnion
    CREDIT_SUMMARY_XPN: Optional[CREDIT_SUMMARY] = None #Experian

# routes
@app.post("/add-user-credit-data")
async def add_user_credit_data(historic_credit:CreditRequest):
        if os.getenv("API_KEY") != historic_credit.API_KEY:
            raise HTTPException(status_code=400, detail="Api key dont match")
        documents = []
         # load personal data
        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.FirstName is None:
            documents.append(
                Document(
                    page_content= f"My firstName: {historic_credit.BORROWER.FirstName}",
                    metadata={"field": "FirstName", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="FirstName",
                )
            )
 
        # load credit inquiry
        if not historic_credit.CREDIT_INQUIRY is None:
            for i in historic_credit.CREDIT_INQUIRY:
                documents.append(Document(
                    page_content= f"Inquiry: {i.Name}; From Credit Repository: {i.CREDIT_REPOSITORY.SourceType}; Date: {i.Date}",
                    metadata={"source": "CreditInquiry", "credit_repository": i.CREDIT_REPOSITORY.SourceType, "date": i.Date, "user_id": historic_credit.USER_ID},
                    id=i.CreditInquiryID,
                ))
        # load credit summary
        if not historic_credit.CREDIT_SUMMARY_EFX is None:
            for i in historic_credit.CREDIT_SUMMARY_EFX.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From Credit Repository: Equifax",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_TUI is None:
            for i in historic_credit.CREDIT_SUMMARY_TUI.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From Credit Repository: TransUnion",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_XPN is None:
            for i in historic_credit.CREDIT_SUMMARY_XPN.DATA_SET:
                documents.append(Document(
                    page_content= f"{i.Name}: {i.Value}. From Credit Repository: Experian",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))

        # load credit score
        if not historic_credit.CREDIT_SCORE is None:
            for i in historic_credit.CREDIT_SCORE:
                credit_repository = i.CreditRepositorySourceType
                date_of_credit_score = i.Date
                documents.append(Document(
                            page_content= f"Credit Score: Value on {date_of_credit_score}:  {i.Value}. From Credit Repository: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=i.Date,
                        ))
                if not i.FACTOR is None:
                    for j in i.FACTOR:
                        documents.append(Document(
                            page_content= f"Credit Score: Negative Factor ({date_of_credit_score}):  {j.Text}. From Credit Repository: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=j.Code,
                        ))
                if not i.POSITIVE_FACTOR is None:
                    for j in i.POSITIVE_FACTOR:
                        documents.append(Document(
                            page_content= f"Credit Score: Positive Factor ({date_of_credit_score}):  {j.Text}. From Credit Repository: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "positive_factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=j.Code,
                        ))

        # load credit liability 
        if not historic_credit.CREDIT_LIABILITY is None:
            for liability in historic_credit.CREDIT_LIABILITY:
                content = f"Credit Loan: "
                content += f"Credit Loan Type: {liability.CreditLoanType}"
                content += f"To Creditor: {liability.CREDITOR.Name}. "
                if not liability.PAYMENT_PATTERN.StartDate is None:
                    content += f"Start date: {liability.PAYMENT_PATTERN.StartDate}. "
                if not liability.OriginalBalanceAmount is None:
                    content += f"Original amount to pay: {liability.OriginalBalanceAmount}. "
                if not liability.UnpaidBalanceAmount is None:
                    content += f"Amount remaining to be paid: {liability.UnpaidBalanceAmount}. "
                
                if not liability.TermsMonthsCount is None:
                    content += f"Total months to pay: {liability.TermsMonthsCount}. "
                elif not liability.TermsDescription is None:
                    content += f"Months/payment description: {liability.TermsDescription}. "
                
                if not liability.MonthsReviewedCount is None:
                    content += f"Months paid so far: {liability.MonthsReviewedCount}. "
                if not liability.HIGHEST_ADVERSE_RATING is None:
                    content += f"Late payment: {liability.HIGHEST_ADVERSE_RATING.Type}. "
                if isinstance(liability.CREDIT_REPOSITORY, list):
                    content += f"From Credit Repository: "
                    for repo in liability.CREDIT_REPOSITORY:
                        content += f"{repo.SourceType}, "
                    content += ". "
                else:
                    content += f"From Credit Repository: {liability.CREDIT_REPOSITORY.SourceType}. "

                documents.append(Document(
                    page_content= content,
                    metadata={"source": "CreditLiability", "field": "liability", "date": liability.PAYMENT_PATTERN.StartDate, "user_id": historic_credit.USER_ID },
                    id=liability.CreditLiabilityID,
                ))
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        return "ok"
  
# model
class Message(BaseModel):
    input: Optional[str] = None
    output: Optional[str] = None

class QueryRequest(BaseModel):
    API_KEY: str
    user_id: str
    query: str
    last_messages: Optional[list[Message]] = None

# endpoint to retrieve an answer
@app.post("/query")
async def query(query_request:QueryRequest):
    if os.getenv("API_KEY") != query_request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2, "k": 20, "filter": {"user_id": query_request.user_id} }
    )
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-pro",
    #     temperature=0,
    # )
    
    memory = "\n".join(
        [f"User: {m.input}\nAssistant: {m.output}" for m in query_request.last_messages]
    )

    llm = ChatOpenAI()

    system_prompt = (
        f"Work as an assistant about credit to myself. "
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Don't give me any advice. "
        "Don't tell me how to repair credit. "
        "Don't tell me how to improve my credit score. "
        "Don't tell me how to write a letter to a collection agency. "
        "Don't answer questions that are not related to credit. "
        "keep the answer concise. "
        "Only respond with the data and some concept related to credit if it necesary. "
        "Previous Messages: " + memory + "\n"
        "Context: {context} "
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    response = chain.invoke({"input": query_request.query})
    return {
        'answer': response['answer'],
    }

# endpoint for knowing when user is on db
class IsUserCreditDataRequest(BaseModel):
    API_KEY: str
    user_id: str

@app.post("/is-user-credit-data")
async def get_is_user_credit_data(request:IsUserCreditDataRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    response = vector_store.get(where={"user_id": request.user_id})
    if len(response['documents']) > 0:
        return "ok"
    else:
        raise HTTPException(status_code=404, detail="This user does'nt have credit data")

class DeleteUserCreditDataRequest(BaseModel):
    API_KEY: str
    user_id: str

# endpoint to delete user credit data
@app.delete("/delete-user-credit-data")
async def delete_user_credit_data(request: DeleteUserCreditDataRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    vector_store.delete(where={"user_id": request.user_id})
    return "ok"

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)