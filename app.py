# imports
from fastapi import FastAPI, Request, HTTPException
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
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn 
import os
import json

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
    PurposeType: Optional[str] = None
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


# load translations
with open("translations.json", "r") as f:
    translations = json.load(f)

def get_translation(text: str) -> str:
    if text in translations:
        return translations[text]
    else:
        return text

def get_credit_liability_total(prefix: str, loan_types: list[str], credit_repositories: list[str], historic_credit:CreditRequest) -> str:
    total_content = f"{prefix} de cuentas de credito: "
    filtered_liabilities = []
    for liability in historic_credit.CREDIT_LIABILITY:
        repo_match = False
        if isinstance(liability.CREDIT_REPOSITORY, list):
            repo_match = any(
                getattr(repo, "SourceType", None) in credit_repositories
                for repo in liability.CREDIT_REPOSITORY
            )
        else:
            repo_match = getattr(liability.CREDIT_REPOSITORY, "SourceType", None) in credit_repositories
        loan_type_match = getattr(liability, "CreditLoanType", None) in loan_types
        if repo_match and loan_type_match:
            filtered_liabilities.append(liability)

    return f"{total_content} {len(filtered_liabilities)}"

def get_credit_cards_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["CreditCard", "ChargeAccount"]
    return get_credit_liability_total("Numero de tarjetas de credito", desired_credit_loan_types, credit_repositories, historic_credit)

def get_auto_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["Automobile"]
    return get_credit_liability_total("Numero de prestamos de auto", desired_credit_loan_types, credit_repositories, historic_credit)

def get_education_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["Educational"]
    return get_credit_liability_total("Numero de prestamos estudiantiles", desired_credit_loan_types, credit_repositories, historic_credit)
    
def get_mortgage_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["ConventionalRealEstateMortgage"]
    return get_credit_liability_total("Numero de prestamos inmobiliarios", desired_credit_loan_types, credit_repositories, historic_credit)

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
                    page_content= f"Mi primer nombre es: {historic_credit.BORROWER.FirstName}",
                    metadata={"field": "FirstName", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="FirstName",
                )
            )
 
        # load credit inquiry
        if not historic_credit.CREDIT_INQUIRY is None:
            for i in historic_credit.CREDIT_INQUIRY:
                documents.append(Document(
                    page_content= f"Consulta: {i.Name}; Tipo de Consulta: {i.PurposeType or "Desconocida"}; Buro de Credito: {i.CREDIT_REPOSITORY.SourceType}; Fecha: {i.Date}",
                    metadata={"source": "CreditInquiry", "credit_repository": i.CREDIT_REPOSITORY.SourceType, "date": i.Date, "user_id": historic_credit.USER_ID},
                    id=i.CreditInquiryID,
                ))
        # load credit summary
        if not historic_credit.CREDIT_SUMMARY_EFX is None:
            for i in historic_credit.CREDIT_SUMMARY_EFX.DATA_SET:
                translated_name = get_translation(i.Name)
                documents.append(Document(
                    page_content= f"{translated_name}: {i.Value} en el Buro de Credito: Equifax",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_TUI is None:
            for i in historic_credit.CREDIT_SUMMARY_TUI.DATA_SET:
                translated_name = get_translation(i.Name)
                documents.append(Document(
                    page_content= f"{translated_name}: {i.Value} en el Buro de Credito: TransUnion",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))
        if not historic_credit.CREDIT_SUMMARY_XPN is None:
            for i in historic_credit.CREDIT_SUMMARY_XPN.DATA_SET:
                translated_name = get_translation(i.Name)
                documents.append(Document(
                    page_content= f"{translated_name}: {i.Value} en el Buro de Credito: Experian",
                    metadata={"source": "CreditSummary", "user_id": historic_credit.USER_ID},
                    id=i.ID,
                ))

        # load credit score
        if not historic_credit.CREDIT_SCORE is None:
            for i in historic_credit.CREDIT_SCORE:
                credit_repository = i.CreditRepositorySourceType
                date_of_credit_score = i.Date
                documents.append(Document(
                            page_content= f"Puntaje de Credito: Valor en la fecha {date_of_credit_score}:  {i.Value} en el Buro de Credito: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=i.Date,
                        ))
                if not i.FACTOR is None:
                    for j in i.FACTOR:
                        translated_text = get_translation(j.Text)
                        documents.append(Document(
                            page_content= f"Puntaje de Credito: Factor Negativo ({date_of_credit_score}):  {translated_text} en el Buro de Credito: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=j.Code,
                        ))
                if not i.POSITIVE_FACTOR is None:
                    for j in i.POSITIVE_FACTOR:
                        translated_text = get_translation(j.Text)
                        documents.append(Document(
                            page_content= f"Puntaje de Credito: Factor Positivo ({date_of_credit_score}):  {translated_text} en el Buro de Credito: {credit_repository}",
                            metadata={"source": "CreditScore", "field": "positive_factor", "date": date_of_credit_score, "user_id": historic_credit.USER_ID, 'credit_repository': credit_repository },
                            id=j.Code,
                        ))

        # load credit liability 
        if not historic_credit.CREDIT_LIABILITY is None:
            for credit_repository in ["TransUnion", "Experian", "Equifax"]:
                documents.append(Document(
                    page_content= f"{get_credit_cards_content(historic_credit, [credit_repository])} en el Buro de Credito: {credit_repository}",
                    metadata={"source": "CreditLiability", "field": "credit_cards", "user_id": historic_credit.USER_ID, "credit_repository": credit_repository },
                    id="CreditCards",
                ))
                documents.append(Document(
                    page_content= f"{get_auto_loans_content(historic_credit, [credit_repository])} en el Buro de Credito: {credit_repository}",
                    metadata={"source": "CreditLiability", "field": "auto_loans", "user_id": historic_credit.USER_ID, "credit_repository": credit_repository },
                    id="AutoLoans",
                ))
                documents.append(Document(
                    page_content= f"{get_education_loans_content(historic_credit, [credit_repository])} en el Buro de Credito: {credit_repository}",
                    metadata={"source": "CreditLiability", "field": "education_loans", "user_id": historic_credit.USER_ID, "credit_repository": credit_repository },
                    id="EducationLoans",
                ))
                documents.append(Document(
                    page_content= f"{get_mortgage_loans_content(historic_credit, [credit_repository])} en el Buro de Credito: {credit_repository}",
                    metadata={"source": "CreditLiability", "field": "mortgage_loans", "user_id": historic_credit.USER_ID, "credit_repository": credit_repository },
                    id="MortgageLoans",
                ))


            for liability in historic_credit.CREDIT_LIABILITY:
                translated_credit_loan_type = get_translation(liability.CreditLoanType)
                content = f"{translated_credit_loan_type}"
                content += f"con el Credor: {liability.CREDITOR.Name}. "
                if not liability.PAYMENT_PATTERN.StartDate is None:
                    content += f"Fecha de inicio: {liability.PAYMENT_PATTERN.StartDate}. "
                if not liability.OriginalBalanceAmount is None:
                    content += f"Monto original a pagar: {liability.OriginalBalanceAmount}. "
                if not liability.UnpaidBalanceAmount is None:
                    content += f"Monto restante a pagar: {liability.UnpaidBalanceAmount}. "
                
                if not liability.TermsMonthsCount is None:
                    content += f"Total de meses a pagar: {liability.TermsMonthsCount}. "
                elif not liability.TermsDescription is None:
                    content += f"Meses/descripcion de pago: {liability.TermsDescription}. "
                
                if not liability.MonthsReviewedCount is None:
                    content += f"Meses pagados hasta la fecha: {liability.MonthsReviewedCount}. "
                if not liability.HIGHEST_ADVERSE_RATING is None:
                    content += f"Pago atrasado: {liability.HIGHEST_ADVERSE_RATING.Type}. "
                if isinstance(liability.CREDIT_REPOSITORY, list):
                    content += f"Buro de Credito: "
                    for repo in liability.CREDIT_REPOSITORY:
                        content += f"{repo.SourceType}, "
                    content += ". "
                else:
                    content += f"Buro de Credito: {liability.CREDIT_REPOSITORY.SourceType}. "

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
        search_kwargs={"score_threshold": 0.2, "k": 30, "filter": {'$or':[{"user_id": query_request.user_id}, {"source": "General Knowledge"}]}  }
    )
    
    memory = "\n".join(
        [f"Usuario: {m.input}\nAsistente: {m.output}" for m in query_request.last_messages]
    )

    llm = ChatOpenAI()

    system_prompt = (
        f"Eres un asistente inteligente dentro de Dumbo Credit."
        "Usa el contexto dado para responder la pregunta. "
        "Si no sabes la respuesta, di que no lo sabes. "
        "Tu única función es ayudar a los usuarios a entender su reporte de crédito y responder preguntas relacionadas con su historial, cuentas, puntaje y datos generales. "
        "No das consejos legales ni ayudas a disputar ni reparar crédito. "
        "Usa un tono profesional pero amigable."
        "Siempre en español claro y sencillo pero con una respuesta bien argumentada. "
        "No uses jerga financiera complicada. "
        "Si una pregunta requiere acción o disputa, responde: En este momento no puedo ayudarte con disputas, pero puedo explicarte qué significa este dato y cómo impacta tu reporte. "
        "Responde siempre que puedas dando datos del reporte de credito. "
        "Si la pregunta es sobre dónde consultar un dato, explica cómo se puede obtener esa información en la vida real, como lo haría una persona fuera del sistema, sin mencionar detalles técnicos, archivos, JSON ni contexto interno."
        "Mensajes Anteriores: " + memory + "."
        "Contexto: {context} "
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

def insert_general_knowledge():
    try:
        vector_store.delete(where={"source": "General Knowledge"})
        documents = [
            Document(
                page_content= f"¿Dónde veo si tengo pagos atrasados?: En la sección de historial de pagos de cada cuenta verás los pagos mes a mes. Si hay un número como “30” o “60”, significa que hubo un pago atrasado de 30 o 60 días en ese mes. Si todo dice “OK”, significa que los pagos han sido puntuales.",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK1",
            ),
            Document(
                page_content= f"¿Dónde puedo simular mi puntaje de credito?: En la sección de simulador de puntaje.",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK2",
            ),
            Document(
                page_content= f"¿Como funciona el simulador de puntaje?: Simulador de Puntaje comienza con la información de su informe de crédito actual y analiza cómo el cambio de esa información podría afectar su puntuación. Por supuesto, todo es hipotético. En realidad, la simulación de estos cambios no afectará su puntaje ni su informe.",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK3",
            ),
            Document(
                page_content= f"¿En que modelo de puntaje se base el simulador de credito?: Nuestra herramienta se basa en los puntajes de crédito de VantageScore® 3.0. Su puntaje siempre cambiará en función del modelo que utilice en ese momento.",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK4",
            ),
            Document(
                page_content= f"¿Donde puedo aprender mas sobre credito?: Tenemos varios cursos en la plataforma para que puedas aprender mas sobre credito. Puede acceder a ellos en la sección de cursos disponible en el menu de usuario.",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK5",
            ),
            Document(
                page_content= f"Rangos de puntaje de credito: 300-579: Muy bajo, 580-669: Regular, 670-739: Bueno, 740-799: Muy bueno, 800+: Excelente",
                metadata={"field": "Website", "source": "General Knowledge" },
                id="GK6",
            ),
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
    except Exception as e:
        print(e)


insert_general_knowledge()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)