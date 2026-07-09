import os
from dotenv import load_dotenv
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# imports
from fastapi import FastAPI, Request, HTTPException, Query, Response
import logging
import threading
from langchain_core.callbacks import BaseCallbackHandler
from fastapi.responses import JSONResponse
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
import uvicorn 
from utils.get_liability_content import get_liability_content
from utils.totals_liabilities import get_credit_cards_content, get_auto_loans_content, get_education_loans_content, get_mortgage_loans_content
from utils.get_translation import get_translation
from utils.get_city_by_code import get_city_by_code
from models import CreditRequest, Lesson, AddLessonRequest, CreditPlan, GeneratePlanRequest, AddUserCreditDataV3Request, CreditReportV3
from utils.get_score_rating import get_score_rating
from utils.build_credit_documents_v3 import build_credit_documents_v3
from utils.crypto import decrypt_body
from utils.prompts import scan_documents, get_disputes_by_pdf_prompt, extract_credit_data_from_pdf_prompt, get_litigation_errors_prompt
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from functools import lru_cache
import hashlib
import time
import tiktoken

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

credit_db_dir = "./credit_db"

client = chromadb.PersistentClient(path=credit_db_dir)

def get_collection_name(user_id:str) -> str:
    return f"{user_id}_credit_collection"

def get_lessons_collection():
    return client.get_or_create_collection(name="lessons_collection")


# %% Caché por reporte de crédito %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Los endpoints de análisis (get-disputes, get-litigation-errors, verify-errors,
# generate-plan) son deterministas respecto al reporte de crédito: mismo reporte
# (y mismos parámetros) -> mismo resultado. Cacheamos por hash del input exacto
# que recibe el LLM para evitar repetir esas llamadas costosas. La clave incluye
# el contenido del reporte, así que cualquier cambio de datos invalida el caché.

def _input_hash(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

def _cache_put(cache: dict, user_id: str, key: str, value):
    # Al cambiar el reporte la clave cambia; eliminamos entradas viejas del mismo
    # usuario para no acumular memoria indefinidamente.
    for k in [k for k in cache if k[0] == user_id]:
        del cache[k]
    cache[(user_id, key)] = value

_disputes_cache: dict = {}      # (user_id, hash) -> list[ErrorDispute]
_litigation_cache: dict = {}    # (user_id, hash) -> list[LitigationError]
_verify_cache: dict = {}        # (user_id, hash) -> VerifyErrorsResponse
_plan_cache: dict = {}          # (user_id, hash) -> (CreditPlan, expiry_ts)
_letters_cache: dict = {}       # (user_id, hash) -> respuesta de generate-letter (idempotencia)
_user_report_hash: dict = {}    # user_id -> hash del contenido ya embebido (idempotencia de add-user-credit-data)


# %% Tracking de consumo de tokens %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Callback que acumula el uso de tokens de TODAS las llamadas al LLM dentro de un
# request (incluso las que usan with_structured_output, que normalmente descartan
# esa info). Es seguro entre hilos (verify-errors corre lotes en paralelo) y entre
# tareas async (get-disputes lanza varias ainvoke concurrentes).

class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.by_model: dict = {}
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        try:
            for generations in response.generations:
                for gen in generations:
                    message = getattr(gen, "message", None)
                    usage = getattr(message, "usage_metadata", None) if message is not None else None
                    if not usage:
                        continue
                    inp = usage.get("input_tokens", 0) or 0
                    out = usage.get("output_tokens", 0) or 0
                    tot = usage.get("total_tokens", 0) or (inp + out)
                    model = (getattr(message, "response_metadata", {}) or {}).get("model_name", "unknown")
                    with self._lock:
                        self.input_tokens += inp
                        self.output_tokens += out
                        self.total_tokens += tot
                        m = self.by_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                        m["input_tokens"] += inp
                        m["output_tokens"] += out
                        m["total_tokens"] += tot
        except Exception as e:
            logger.warning(f"No se pudo registrar el uso de tokens: {e}")

def _set_usage_headers(response: Response, tracker: "TokenUsageTracker"):
    response.headers["X-Usage-Input-Tokens"] = str(tracker.input_tokens)
    response.headers["X-Usage-Output-Tokens"] = str(tracker.output_tokens)
    response.headers["X-Usage-Total-Tokens"] = str(tracker.total_tokens)
    response.headers["X-Usage-By-Model"] = json.dumps(tracker.by_model)

# --- Uso de tokens de EMBEDDINGS -------------------------------------------
# Los embeddings de OpenAI sí consumen tokens (solo de entrada), pero LangChain
# no expone ese uso por callbacks. Los calculamos localmente con tiktoken; el
# conteo coincide con el tokenizador que usa OpenAI para el modelo de embeddings.
EMBEDDING_MODEL = "text-embedding-3-large"
try:
    _embed_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
except Exception:
    _embed_encoding = tiktoken.get_encoding("cl100k_base")

def _track_embedding_usage(tracker: "TokenUsageTracker", texts) -> int:
    """Suma al tracker los tokens de entrada que costará embeber `texts`."""
    n = 0
    for t in texts:
        if t:
            n += len(_embed_encoding.encode(t))
    with tracker._lock:
        tracker.input_tokens += n
        tracker.total_tokens += n
        m = tracker.by_model.setdefault(EMBEDDING_MODEL, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        m["input_tokens"] += n
        m["total_tokens"] += n
    return n

# Helper function para obtener vector store (reutiliza conexiones)
@lru_cache(maxsize=100)
def get_vector_store(user_id: str) -> Chroma:
    """
    Obtiene o crea una instancia de Chroma para un usuario específico.
    Utiliza lru_cache para evitar la re-instanciación costosa.
    """
    return Chroma(
        collection_name=f"{user_id}_credit_collection",
        embedding_function=embeddings,
        persist_directory=credit_db_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _embed_and_store_documents(user_id: str, documents: list, tracker: "TokenUsageTracker", response: Response) -> str:
    """Reembebe (con idempotencia por hash) la colección de crédito del usuario.

    Comparte la lógica de idempotencia/borrado/embedding de /add-user-credit-data:
    si el contenido es idéntico al ya indexado no vuelve a embeber (el refresco hace
    polling y reenvía los mismos datos varias veces, y re-embeber cuesta decenas de
    miles de tokens cada vez)."""
    content_hash = _input_hash(str(len(documents)), "\x00".join(d.page_content for d in documents))
    if documents and _user_report_hash.get(user_id) == content_hash:
        return "ok"

    # Optimización: intentar borrar directamente sin listar todas las colecciones.
    try:
        client.delete_collection(name=get_collection_name(user_id))
        get_vector_store.cache_clear()
    except Exception:
        pass

    uuids = [get_collection_name(user_id) + str(uuid4()) for _ in range(len(documents))]
    vector_store = get_vector_store(user_id)
    _track_embedding_usage(tracker, [d.page_content for d in documents])
    vector_store.add_documents(documents=documents, ids=uuids)
    if documents:
        _user_report_hash[user_id] = content_hash

    # Solo emitimos headers (y registramos consumo) si de verdad se embebió algo.
    if tracker.total_tokens > 0:
        _set_usage_headers(response, tracker)
    return "ok"


# env vars

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

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc),
            "path": request.url.path
        }
    )


# routes
@app.post("/add-user-credit-data")
async def add_user_credit_data(historic_credit:CreditRequest, response: Response):
        if os.getenv("API_KEY") != historic_credit.API_KEY:
            raise HTTPException(status_code=400, detail="Api key dont match")
        tracker = TokenUsageTracker()
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
        
        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.MiddleName is None:
            documents.append(
                Document(
                    page_content= f"Mi segundo nombre es: {historic_credit.BORROWER.MiddleName}",
                    metadata={"field": "MiddleName", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="MiddleName",
                )
            )
        
        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.LastName is None:
            documents.append(
                Document(
                    page_content= f"Mi apellido es: {historic_credit.BORROWER.LastName}",
                    metadata={"field": "LastName", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="LastName",
                )
            )

        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.SSN is None:
            documents.append(
                Document(
                    page_content= f"Mi SSN es: {historic_credit.BORROWER.SSN}",
                    metadata={"field": "SSN", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="SSN",
                )
            )

        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.BirthDate is None:
            documents.append(
                Document(
                    page_content= f"Mi Fecha de nacimiento es: {historic_credit.BORROWER.BirthDate}",
                    metadata={"field": "BirthDate", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="BirthDate",
                )
            )

        if not historic_credit.BORROWER is None and not historic_credit.BORROWER.RESIDENCE is None:
            addresses = f"Mis direcciones son: "
            if type(historic_credit.BORROWER.RESIDENCE) == list:
                for residence in historic_credit.BORROWER.RESIDENCE: 
                    address = ""
                    if not residence.BorrowerResidencyType is None:
                        if residence.BorrowerResidencyType == "Current":
                            address += "Residiendo Actualmente en: "
                        elif residence.BorrowerResidencyType == "Prior":
                            address += "Anteriormente residiendo en: "
                        else: 
                            address += f"Residencia ({residence.BorrowerResidencyType}) en: "
                    if not residence.City is None:
                        address += f"Ciudad {residence.City}, "
                    if not residence.State is None:
                        address += f"Estado {residence.State}, "
                    if not residence.PostalCode is None:
                        address += f"Codigo Postal {residence.PostalCode}, "
                    if not residence.StreetAddress is None:
                        address += f"Calle {residence.StreetAddress}, "
                    address = address[:-2]
                    address += "; "

                    addresses += address
            else:
                address = ""
                if not historic_credit.BORROWER.RESIDENCE.BorrowerResidencyType is None:
                    if historic_credit.BORROWER.RESIDENCE.BorrowerResidencyType == "Current":
                        address += "Residiendo Actualmente en: "
                    elif historic_credit.BORROWER.RESIDENCE.BorrowerResidencyType == "Prior":
                        address += "Anteriormente residiendo en: "
                    else: 
                        address += f"Residencia ({historic_credit.BORROWER.RESIDENCE.BorrowerResidencyType}) en: "
                if not historic_credit.BORROWER.RESIDENCE.City is None:
                    address += f"Ciudad {historic_credit.BORROWER.RESIDENCE.City}, "
                if not historic_credit.BORROWER.RESIDENCE.State is None:
                    address += f"Estado {historic_credit.BORROWER.RESIDENCE.State}, "
                if not historic_credit.BORROWER.RESIDENCE.PostalCode is None:
                    address += f"Codigo Postal {historic_credit.BORROWER.RESIDENCE.PostalCode}, "
                if not historic_credit.BORROWER.RESIDENCE.StreetAddress is None:
                    address += f"Calle {historic_credit.BORROWER.RESIDENCE.StreetAddress}, "
                address = address[:-2]
                address += "; "
                addresses += address

            documents.append(
                Document(
                    page_content= addresses,
                    metadata={"field": "Address", "source": "Personal Info", "user_id": historic_credit.USER_ID },
                    id="Address",
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
                if not i.Value is None:
                    documents.append(Document(
                                page_content= f"Puntaje de Credito: Valor en la fecha {date_of_credit_score}:  {i.Value} en el Buro de Credito: {credit_repository}, clasificacion: {get_score_rating(i.Value)}",
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
                content = get_liability_content(liability)
                if liability.PAYMENT_PATTERN:
                    documents.append(Document(
                        page_content= content,
                        metadata={"source": "CreditLiability", "field": "liability", "date": liability.PAYMENT_PATTERN.StartDate, "user_id": historic_credit.USER_ID },
                        id=liability.CreditLiabilityID,
                    ))
                else:
                    documents.append(Document(
                        page_content= content,
                        metadata={"source": "CreditLiability", "field": "liability", "user_id": historic_credit.USER_ID },
                        id=liability.CreditLiabilityID,
                    ))


        # Idempotencia: si el contenido del reporte es idéntico al ya indexado, NO
        # re-embebemos (el flujo de refresco hace polling y reenvía los mismos datos
        # varias veces; re-embeber cuesta decenas de miles de tokens cada vez).
        content_hash = _input_hash(str(len(documents)), "\x00".join(d.page_content for d in documents))
        if documents and _user_report_hash.get(historic_credit.USER_ID) == content_hash:
            return "ok"

        # Optimización: Intentar borrar directamente sin listar todas las colecciones
        try:
            client.delete_collection(name=get_collection_name(historic_credit.USER_ID))
            # Limpiar el caché de la instancia borrada
            get_vector_store.cache_clear()
        except Exception:
            # Si no existe, no hacemos nada
            pass


        uuids = [get_collection_name(historic_credit.USER_ID) + str(uuid4()) for _ in range(len(documents))]
        vector_store = get_vector_store(historic_credit.USER_ID)
        _track_embedding_usage(tracker, [d.page_content for d in documents])
        vector_store.add_documents(documents=documents, ids=uuids)
        if documents:
            _user_report_hash[historic_credit.USER_ID] = content_hash

        # Solo emitimos headers (y por tanto registramos consumo) si de verdad se
        # embebió algo. Sin datos -> 0 tokens, pero NO es un cache hit, así que no
        # lo reportamos para no marcarlo como "cache".
        if tracker.total_tokens > 0:
            _set_usage_headers(response, tracker)
        return "ok"


@app.post("/add-user-credit-data-v3")
async def add_user_credit_data_v3(request: AddUserCreditDataV3Request, response: Response):
    """Igual que /add-user-credit-data pero con la estructura nueva (CreditReport 3B
    de Equifax). El reporte llega cifrado en `data` (AES-256-GCM, esquema de dumbo-prod);
    aquí se descifra, se convierte a Documents y se embebe en el vector store del usuario."""
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    try:
        raw_report = decrypt_body(request.data)
    except Exception as e:
        logger.error(f"add-user-credit-data-v3: no se pudo descifrar el reporte: {e}")
        raise HTTPException(status_code=400, detail="Could not decrypt credit data")

    try:
        report = CreditReportV3.model_validate(raw_report)
    except Exception as e:
        logger.error(f"add-user-credit-data-v3: reporte con formato invalido: {e}")
        raise HTTPException(status_code=422, detail="Invalid credit report format")

    tracker = TokenUsageTracker()
    documents = build_credit_documents_v3(request.USER_ID, report)
    return _embed_and_store_documents(request.USER_ID, documents, tracker, response)


# model
class Message(BaseModel):
    input: Optional[str] = None
    output: Optional[str] = None

class QueryRequest(BaseModel):
    API_KEY: str
    user_id: str
    query: str
    last_messages: Optional[list[Message]] = None

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

class QueryResponse(BaseModel):
    answer: str

# endpoint to retrieve an answer
@app.post("/query")
async def query(query_request:QueryRequest, response: Response) -> QueryResponse:
    if os.getenv("API_KEY") != query_request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    vector_store = get_vector_store(query_request.user_id)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2, "k": 30, "filter": {'$or':[{"user_id": query_request.user_id}, {"source": "General Knowledge"}]}  }
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
        "Siempre que puedas, responde diferenciando la información por cada buró: TransUnion, Equifax y Experian. "
        "Los datos no se suman entre los burós, ya que la misma tarjeta de crédito u otra cuenta puede estar reportada en los tres burós al mismo tiempo, y lo mismo aplica para otros datos. "
        "Una misma cuenta puede aparecer en los tres reportes, pero no es que sean cuentas diferentes ni que se sumen los montos. "
        "Para responder sobre el puntaje ten en cuenta siempre esta unica escala de puntaje, no otra, la escala es: de 300 hasta 579 para Muy bajo, de 580 hasta 669 para Regular, de 670 hasta 739 para Bueno, de 740 hasta 799 para Muy bueno, y de 800 en adelante para Excelente. "
        "Si una pregunta requiere acción o disputa, responde solamente, sin dar consejos legales ni ayudas a disputar ni reparar crédito: En este momento no puedo ayudarte con disputas, pero puedo explicarte qué significa tus datos y cómo impactan en tu reporte. "
        "No respondas preguntas que no sean relacionadas con el credito. "
        "No respondas preguntas sobre como reparar el credito, solo responde que no puedo ayudarte con eso. "
        "No respondas preguntas sobre como mejorar el puntaje. "
        "No des ningun consejo o instruccion acerca de como aumentar ni llevar de una calificacion a otra el puntaje. " 
        "No respondas como convertir el puntaje a excelente, bueno, regular, etc. "
        "No respondas preguntas sobre como hacer cartas de disputa, su estructura, la información que debe contener, etc. "
        "Responde siempre que puedas dando datos del reporte de credito. "
        "Si la pregunta es sobre dónde consultar un dato, explica cómo se puede obtener esa información en la vida real, como lo haría una persona fuera del sistema, sin mencionar detalles técnicos, archivos, JSON ni contexto interno."
        "Contexto: {context} "
    )

    memory = [("system", system_prompt)] 
    
    for m in query_request.last_messages:
        memory.append(("human", m.input))
        memory.append(("ai", m.output))

    memory.append(("human", "{input}"))

    prompt = ChatPromptTemplate(memory)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    chain_response = chain.invoke({"input": query_request.query}, config={"callbacks": [tracker]})
    _set_usage_headers(response, tracker)
    return {
        'answer': chain_response['answer'],
    }

class AiAnswer(BaseModel):
    answer: str = Field(description="Respuesta");
    must_talk_with_a_human: bool = Field(description="Si el usuario debe contactar o no con un humano para la pregunta que esta haciendo");

class PosAiAnswer(BaseModel):
    must_talk_with_a_human: bool = Field(description="Si el usuario debe contactar o no con un humano para la pregunta que esta haciendo");

@app.post("/query-without-limits")
async def query_without_limits(query_request:QueryRequest, response: Response) -> AiAnswer:
    if os.getenv("API_KEY") != query_request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    vector_store = get_vector_store(query_request.user_id)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2, "k": 30, "filter": {'$or':[{"user_id": query_request.user_id}, {"source": "General Knowledge"}]}  }
    )
    
    llm = ChatOpenAI()

    system_prompt = (
        f"Eres un asistente inteligente dentro de Dumbo Credit."
        "Usa el contexto dado para responder la pregunta. "
        "Si no sabes la respuesta, di que no lo sabes. "
        "Tu única función es ayudar a los usuarios a entender su reporte de crédito y responder preguntas relacionadas con su historial, cuentas, puntaje y datos generales. "
        "Puedes ayudar al usuario a elavorar una carta de disputa, su estructura, la información que debe contener, etc. "
        "Puedes darle consejos sobre como mejorar el puntaje, como aumentar el puntaje, como llevar de una calificacion a otra el puntaje, etc. "
        "Usa un tono profesional pero amigable."
        "Siempre en español claro y sencillo pero con una respuesta bien argumentada. "
        "No uses jerga financiera complicada. "
        "Siempre que puedas, responde diferenciando la información por cada buró: TransUnion, Equifax y Experian. "
        "Los datos no se suman entre los burós, ya que la misma tarjeta de crédito u otra cuenta puede estar reportada en los tres burós al mismo tiempo, y lo mismo aplica para otros datos. "
        "Una misma cuenta puede aparecer en los tres reportes, pero no es que sean cuentas diferentes ni que se sumen los montos. "
        "Para responder sobre el puntaje ten en cuenta siempre esta unica escala de puntaje, no otra, la escala es: de 300 hasta 579 para Muy bajo, de 580 hasta 669 para Regular, de 670 hasta 739 para Bueno, de 740 hasta 799 para Muy bueno, y de 800 en adelante para Excelente. "
        "No respondas preguntas que no sean relacionadas con el credito. "
        "Responde siempre que puedas dando datos del reporte de credito. "
        "Si la pregunta es sobre dónde consultar un dato, explica cómo se puede obtener esa información en la vida real, como lo haría una persona fuera del sistema, sin mencionar detalles técnicos, archivos, JSON ni contexto interno."
        "Contexto: {context} "
    )

    memory = [("system", system_prompt)] 
    
    for m in query_request.last_messages:
        memory.append(("human", m.input))
        memory.append(("ai", m.output))

    memory.append(("human", "{input}"))

    prompt = ChatPromptTemplate(memory)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    chain_response = chain.invoke({"input": query_request.query}, config={"callbacks": [tracker]})

    llm = ChatOpenAI()

    structured_llm = llm.with_structured_output(PosAiAnswer)

    prompts = [
        SystemMessage("Identifica si es indispensable la intervencion de una persona humana en este contexto."),
    ]

    for m in query_request.last_messages:
        prompts.append(HumanMessage(content=m.input))
        prompts.append(AIMessage(content=m.output))

    prompts.append(HumanMessage(content=query_request.query))

    pos_response = structured_llm.invoke(prompts, config={"callbacks": [tracker]})

    _set_usage_headers(response, tracker)
    return {
        'answer': chain_response['answer'],
        'must_talk_with_a_human': pos_response.must_talk_with_a_human
    }


# endpoint for knowing when user is on db
class IsUserCreditDataRequest(BaseModel):
    API_KEY: str
    user_id: str

@app.post("/is-user-credit-data")
async def get_is_user_credit_data(request:IsUserCreditDataRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    # Optimización: Acceso directo a la colección sin listar todas (O(1) vs O(N))
    try:
        collection = client.get_collection(name=get_collection_name(request.user_id))
        docs = collection.get(where={"user_id": request.user_id}, limit=1)
        if len(docs['documents']) > 0:
            return "ok"
    except (ValueError, Exception):
        pass

    raise HTTPException(status_code=404, detail="This user does'nt have credit data")

class DeleteUserCreditDataRequest(BaseModel):
    API_KEY: str
    user_id: str

# endpoint to delete user credit data
@app.post("/delete-user-credit-data")
async def delete_user_credit_data(request: DeleteUserCreditDataRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    
    try:
        client.delete_collection(name=get_collection_name(request.user_id))
        get_vector_store.cache_clear()
        _user_report_hash.pop(request.user_id, None)
    except Exception:
        pass

    return "ok"

class DocumentType(str, Enum):
    driver_license = "Licencia de conducir"
    state_issued_id = "Identificación emitida por el Estado"
    passport = "Pasaporte"
    utility_bill = "Factura de servicios públicos"
    bank_account_statement = "Estado de cuenta bancaria"
    lease = "Contrato de arrendamiento"
    letter = "Carta"
    SSN = "SSN"
    misc = "Otro"

class GenericDocumentField(BaseModel):
    field: str = Field(description="Nombre del campo en espaniol");
    value: str = Field(description="Valor del campo, si es una fecha en formato yyyy-mm-dd");

class GenericDocumentData(BaseModel):
    fields: list[GenericDocumentField] = Field(description="Lista de campos y valores del documento");

class DocumentData(BaseModel):
    data: GenericDocumentData = Field(description="Datos del documento");
    is_valid: bool = Field(description="Si el documento es valido");
    error: Optional[str] = Field(description="Mensaje de error si el documento no es valido");
    type: DocumentType = Field(description="Tipo de documento");

class ScanImageRequest(BaseModel):
    API_KEY: str
    image_url: Union[str, list[str]]

@app.post("/scan_image")
async def scan_image(request: ScanImageRequest, response: Response) -> DocumentData:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    tracker = TokenUsageTracker()
    vision_model = ChatOpenAI(model='gpt-4o')
    content = [
            {
                'type': 'text',
                'text': 'A partir de la imagen extrae los datos de la misma, el tipo de documento y si es una imagen valida de un documento. Identifica si tiene algun error esta carta, q permita abrir un caso de litigacion, como puede ser un error de forma, contenido, lenguaje ilegal, amenazas, etc.'
            }
        ]
    if type(request.image_url) == list:
        for image in request.image_url:
            content.append({ 
            'type': 'image_url', 
            'image_url': { 'url': image, 'detail': 'auto'} 
        })
    else: 
        content.append({ 
            'type': 'image_url', 
            'image_url': { 'url': request.image_url, 'detail': 'auto'} 
        })
    prompts = [
        SystemMessage(scan_documents),
        HumanMessage(content=content)
    ]
    structured_llm = vision_model.with_structured_output(DocumentData)
    vision_response = structured_llm.invoke(prompts, config={"callbacks": [tracker]})
    _set_usage_headers(response, tracker)
    return vision_response

# paraphrase letter in english
class ParaphraseLetterResponse(BaseModel):
    paraphrased_letter: str

class ParaphraseLetterRequest(BaseModel):
    API_KEY: str
    letter: str

@app.post("/paraphrase-letter")
async def paraphrase_letter(request: ParaphraseLetterRequest, response: Response):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    tracker = TokenUsageTracker()
    llm = ChatOpenAI()
    prompts = [
        SystemMessage("""
                      Your task is to paraphrase this letter, keeping professional tone and style.  
                      Return the paraphrased letter in english. 
                """),
        HumanMessage(content=[
            {
                'type': 'text',
                'text': 'Paraphrase this letter in english, keeping professional tone and style.'
            },
            { 
                'type': 'text', 
                'text': request.letter
            }
        ] )
    ]
    structured_llm = llm.with_structured_output(ParaphraseLetterResponse)
    result = structured_llm.invoke(prompts, config={"callbacks": [tracker]})
    _set_usage_headers(response, tracker)
    return result

# def insert_general_knowledge():
#     try:
#         vector_store.delete(where={"source": "General Knowledge"})
#         documents = [
#             Document(
#                 page_content= f"¿Dónde veo si tengo pagos atrasados?: En la sección de historial de pagos de cada cuenta verás los pagos mes a mes. Si hay un número como “30” o “60”, significa que hubo un pago atrasado de 30 o 60 días en ese mes. Si todo dice “OK”, significa que los pagos han sido puntuales.",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK1",
#             ),
#             Document(
#                 page_content= f"¿Dónde puedo simular mi puntaje de credito?: En la sección de simulador de puntaje.",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK2",
#             ),
#             Document(
#                 page_content= f"¿Como funciona el simulador de puntaje?: Simulador de Puntaje comienza con la información de su informe de crédito actual y analiza cómo el cambio de esa información podría afectar su puntuación. Por supuesto, todo es hipotético. En realidad, la simulación de estos cambios no afectará su puntaje ni su informe.",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK3",
#             ),
#             Document(
#                 page_content= f"¿En que modelo de puntaje se base el simulador de credito?: Nuestra herramienta se basa en los puntajes de crédito de VantageScore® 3.0. Su puntaje siempre cambiará en función del modelo que utilice en ese momento.",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK4",
#             ),
#             Document(
#                 page_content= f"¿Donde puedo aprender mas sobre credito?: Tenemos varios cursos en la plataforma para que puedas aprender mas sobre credito. Puede acceder a ellos en la sección de cursos disponible en el menu de usuario.",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK5",
#             ),
#             Document(
#                 page_content= f"Rangos de puntaje de credito: 300-579: Muy bajo, 580-669: Regular, 670-739: Bueno, 740-799: Muy bueno, 800+: Excelente",
#                 metadata={"field": "Website", "source": "General Knowledge" },
#                 id="GK6",
#             ),
#         ]
#         uuids = [str(uuid4()) for _ in range(len(documents))]
#         vector_store.add_documents(documents=documents, ids=uuids)
#     except Exception as e:
#         print(e)

class ErrorTypeEnum(str, Enum):
    COLLECTION = "Collection"
    CHARGE_OFF = "Charge off"
    REPOSSESSION = "Repossession"

class Address(BaseModel):
    company_name: str = Field(description="The name of the company or creditor")
    address: str = Field(description="The address of the creditor")
    city: str = Field(description="The city of the creditor")
    state: str = Field(description="The state of the creditor (two letter code)")
    zip_code: str = Field(description="The zip code of the creditor")


class CreditBureauEnum(str, Enum):
    EQUIFAX = "Equifax"
    EXPERIAN = "Experian"
    TRANSUNION = "TransUnion"


class AccountStatusEnum(str, Enum):
    OPEN = "Open"
    CLOSED = "Closed"
    CURRENT = "Current"
    PAID = "Paid"
    PAYS_AS_AGREED = "Pays as agreed"
    CHARGE_OFF = "Charge off"
    COLLECTION = "Collection"
    DELINQUENT = "Delinquent"
    LATE = "Late"
    FORECLOSURE = "Foreclosure"
    REPOSSESSION = "Repossession"
    VOLUNTARY_SURRENDER = "Voluntary surrender"
    SETTLED = "Settled"
    OTHER = "Other"


class PdfPersonalResidence(BaseModel):
    BorrowerResidencyType: Optional[str] = Field(default=None, description="Current, Prior, etc.")
    StreetAddress: Optional[str] = None
    City: Optional[str] = None
    State: Optional[str] = None
    PostalCode: Optional[str] = None


class PdfPersonalInfoFromImage(BaseModel):
    bureau: CreditBureauEnum = Field(
        description="Buró del que proviene este bloque (nombre, SSN, direcciones, etc.)"
    )
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    ssn: Optional[str] = None
    birth_date: Optional[str] = None
    residences: list[PdfPersonalResidence] = Field(default_factory=list)


class PdfAccountFromImage(BaseModel):
    name: Optional[str] = Field(default=None, description="Nombre de la cuenta / tradeline como en el informe")
    account_number: Optional[str] = None
    credit_limit: Optional[float] = None
    balance: Optional[float] = None
    credit_utilization_percent: Optional[float] = Field(
        default=None,
        description="Porcentaje del límite de crédito utilizado (0–100); omitir si no figura o no aplica",
        ge=0,
        le=100,
    )
    monthly_payment: Optional[float] = None
    account_status: Optional[AccountStatusEnum] = Field(
        default=None,
        description="Estado de la cuenta según el informe; usar Other si no encaja en otro valor",
    )
    loan_type: Optional[str] = None
    opened_date: Optional[str] = None
    last_activity: Optional[str] = None
    creditor: Address = Field(description="Acreedor con formato Address")
    bureau: Union[CreditBureauEnum, list[CreditBureauEnum]] = Field(
        description="Buró o burós donde figura esta cuenta (Equifax, Experian, TransUnion)"
    )


class PdfInquiryFromImage(BaseModel):
    name: str
    date: str
    purpose_type: Optional[str] = None
    bureau: CreditBureauEnum = Field(description="Buró donde figura la consulta")


class PdfPublicRecordFromImage(BaseModel):
    record_type: str
    filed_date: Optional[str] = None
    description: Optional[str] = None
    court: Optional[str] = None
    bureau: Optional[CreditBureauEnum] = None


class PdfCreditScoreFromImage(BaseModel):
    bureau: CreditBureauEnum
    date: Optional[str] = None
    value: Optional[int] = None


class CreditDataExtractedFromPdf(BaseModel):
    personal_info: list[PdfPersonalInfoFromImage] = Field(
        default_factory=list,
        description="Un bloque por buró cuando el informe separa la información personal por sección",
    )
    accounts: list[PdfAccountFromImage] = Field(default_factory=list)
    inquiries: list[PdfInquiryFromImage] = Field(default_factory=list)
    public_records: list[PdfPublicRecordFromImage] = Field(default_factory=list)
    credit_scores: list[PdfCreditScoreFromImage] = Field(default_factory=list)


class ReasoningEffortEnum(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AddUserCreditDataByPdfRequest(BaseModel):
    API_KEY: str
    user_id: Optional[str] = None
    image_url: Union[str, list[str]]
    file_id: str
    reasoning_effort: ReasoningEffortEnum = Field(
        default=ReasoningEffortEnum.NONE, description="Nivel de razonamiento del modelo"
    )


def _pdf_personal_info_to_documents(user_id: str, infos: list[PdfPersonalInfoFromImage], file_id: str) -> list[Document]:
    """Mismo formato de texto que /add-user-credit-data para que /generate-letter y el retriever lo encuentren."""
    documents: list[Document] = []
    if not infos:
        return documents
    block = infos[0]
    for candidate in infos:
        if (
            candidate.first_name
            or candidate.last_name
            or candidate.ssn
            or candidate.birth_date
            or candidate.residences
        ):
            block = candidate
            break

    if block.first_name:
        documents.append(
            Document(
                page_content=f"Mi primer nombre es: {block.first_name}",
                metadata={"field": "FirstName", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    if block.middle_name:
        documents.append(
            Document(
                page_content=f"Mi segundo nombre es: {block.middle_name}",
                metadata={"field": "MiddleName", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    if block.last_name:
        documents.append(
            Document(
                page_content=f"Mi apellido es: {block.last_name}",
                metadata={"field": "LastName", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    if block.ssn:
        documents.append(
            Document(
                page_content=f"Mi SSN es: {block.ssn}",
                metadata={"field": "SSN", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    if block.birth_date:
        documents.append(
            Document(
                page_content=f"Mi Fecha de nacimiento es: {block.birth_date}",
                metadata={"field": "BirthDate", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    if block.residences:
        parts: list[str] = []
        for residence in block.residences:
            address = ""
            if residence.BorrowerResidencyType is not None:
                br = residence.BorrowerResidencyType
                if br == "Current":
                    address += "Residiendo Actualmente en: "
                elif br == "Prior":
                    address += "Anteriormente residiendo en: "
                else:
                    address += f"Residencia ({br}) en: "
            if residence.City is not None:
                address += f"Ciudad {residence.City}, "
            if residence.State is not None:
                address += f"Estado {residence.State}, "
            if residence.PostalCode is not None:
                address += f"Codigo Postal {residence.PostalCode}, "
            if residence.StreetAddress is not None:
                address += f"Calle {residence.StreetAddress}, "
            if len(address) >= 2 and address.endswith(", "):
                address = address[:-2]
            address += "; "
            parts.append(address)
        addresses = "Mis direcciones son: " + "".join(parts)
        documents.append(
            Document(
                page_content=addresses,
                metadata={"field": "Address", "source": "Personal Info", "user_id": user_id, "file_id": file_id},
            )
        )
    return documents


def _pdf_bureau_values(bureau: Union[CreditBureauEnum, list[CreditBureauEnum]]) -> list[str]:
    if isinstance(bureau, list):
        return [b.value for b in bureau]
    return [bureau.value]


def _delete_user_credit_documents_keep_general_knowledge(user_id: str, file_id: str) -> None:
    """Quita los documentos de crédito del usuario sin borrar entradas de conocimiento general."""
    collection_name = get_collection_name(user_id)
    if collection_name not in [c.name for c in client.list_collections()]:
        return
    coll = client.get_collection(name=collection_name)
    try:
        res = coll.get(where={"user_id": user_id, "file_id": file_id}, include=["metadatas"])
    except Exception as e:
        logger.warning("No se pudieron listar documentos del usuario para borrar: %s", e)
        return
    ids_list = res.get("ids") or []
    metas = res.get("metadatas") or []
    to_delete: list[str] = []
    for idx, doc_id in enumerate(ids_list):
        meta = metas[idx] if idx < len(metas) else None
        if meta is None:
            to_delete.append(doc_id)
            continue
        if meta.get("source") == "General Knowledge":
            continue
        to_delete.append(doc_id)
    if to_delete:
        coll.delete(ids=to_delete)


def _pdf_account_to_page_content(acc: PdfAccountFromImage) -> str:
    """Texto alineado con get_liability_content para cuentas extraídas del PDF."""
    label = acc.loan_type or "Cuenta"
    name = acc.name or "sin nombre"
    content = f"{label}: {name}. "
    if acc.account_status is not None:
        content += f"Estado de la cuenta: {acc.account_status.value}. "
    if acc.balance is not None:
        content += f"Saldo: {acc.balance}. "
    if acc.credit_limit is not None:
        content += f"Limite crediticio: {acc.credit_limit}. "
    if acc.credit_utilization_percent is not None:
        content += f"Utilizacion del limite: {acc.credit_utilization_percent}%. "
    if acc.monthly_payment is not None:
        content += f"Cantidad de Pago Mensual: {acc.monthly_payment}. "
    if acc.account_number:
        content += f"Numero de cuenta: {acc.account_number}. "
    if acc.opened_date:
        content += f"Fecha de apertura de la cuenta: {acc.opened_date}. "
    if acc.last_activity:
        content += f"Fecha de ultima actividad: {acc.last_activity}. "
    c = acc.creditor
    if c.company_name:
        content += f"Nombre del acreedor: {c.company_name}. "
    if c.address:
        content += f"Direccion del acreedor: {c.address}. "
    if c.city:
        content += f"Ciudad del acreedor: {c.city}. "
    if c.state:
        content += f"Estado del acreedor: {c.state}. "
    if c.zip_code:
        content += f"Codigo postal del acreedor: {c.zip_code}. "
    repos = _pdf_bureau_values(acc.bureau)
    if len(repos) == 1:
        content += f"Buro de Credito: {repos[0]}. "
    else:
        content += f"En los buros de credito: {', '.join(repos)}. "
    return content


def credit_extracted_from_pdf_to_documents(user_id: str, extracted: CreditDataExtractedFromPdf, file_id: str) -> list[Document]:
    documents: list[Document] = []
    documents.extend(_pdf_personal_info_to_documents(user_id, extracted.personal_info, file_id))

    for inq in extracted.inquiries:
        purpose = inq.purpose_type or "Desconocida"
        repo = inq.bureau.value
        documents.append(
            Document(
                page_content=(
                    f"Consulta: {inq.name}; Tipo de Consulta: {purpose}; "
                    f"Buro de Credito: {repo}; Fecha: {inq.date}"
                ),
                metadata={
                    "source": "CreditInquiry",
                    "credit_repository": repo,
                    "date": inq.date,
                    "user_id": user_id,
                    "file_id": file_id,
                },
            )
        )

    for score in extracted.credit_scores:
        if score.value is None:
            continue
        date_of = score.date or ""
        repo = score.bureau.value
        rating = get_score_rating(score.value)
        documents.append(
            Document(
                page_content=(
                    f"Puntaje de Credito: Valor en la fecha {date_of}:  {score.value} "
                    f"en el Buro de Credito: {repo}, clasificacion: {rating}"
                ),
                metadata={
                    "source": "CreditScore",
                    "field": "factor",
                    "date": date_of or "unknown",
                    "user_id": user_id,
                    "credit_repository": repo,
                    "file_id": file_id,
                },
            )
        )

    for rec in extracted.public_records:
        bureau_str = rec.bureau.value if rec.bureau is not None else "Desconocido"
        parts = [f"Registro publico: {rec.record_type}; Buro de Credito: {bureau_str}."]
        if rec.filed_date:
            parts.append(f"Fecha de presentacion: {rec.filed_date}.")
        if rec.description:
            parts.append(f"Descripcion: {rec.description}.")
        if rec.court:
            parts.append(f"Tribunal: {rec.court}.")
        documents.append(
            Document(
                page_content=" ".join(parts),
                metadata={
                    "source": "PublicRecord",
                    "user_id": user_id,
                    "credit_repository": bureau_str,
                    "file_id": file_id,
                },
            )
        )

    for acc in extracted.accounts:
        page = _pdf_account_to_page_content(acc)
        repos = _pdf_bureau_values(acc.bureau)
        primary_repo = repos[0] if repos else "Equifax"
        date_meta = acc.opened_date or acc.last_activity or ""
        meta: dict = {
            "source": "CreditLiability",
            "field": "liability",
            "user_id": user_id,
            "credit_repository": primary_repo,
            "file_id": file_id,
        }
        if date_meta:
            meta["date"] = date_meta
        documents.append(Document(page_content=page, metadata=meta))

    return documents


def persist_pdf_extract_to_vector_store(user_id: str, extracted: CreditDataExtractedFromPdf, file_id: str, tracker: "TokenUsageTracker" = None) -> None:
    _delete_user_credit_documents_keep_general_knowledge(user_id, file_id)
    documents = credit_extracted_from_pdf_to_documents(user_id, extracted, file_id)
    if not documents:
        return
    collection_name = get_collection_name(user_id)
    uuids = [collection_name + str(uuid4()) for _ in range(len(documents))]
    vector_store = get_vector_store(user_id)
    if tracker is not None:
        _track_embedding_usage(tracker, [d.page_content for d in documents])
    vector_store.add_documents(documents=documents, ids=uuids)


@app.post("/add-user-credit-data-by-pdf")
async def add_user_credit_data_by_pdf(
    request: AddUserCreditDataByPdfRequest,
    response: Response,
) -> CreditDataExtractedFromPdf:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    content: list[dict] = []
    if isinstance(request.image_url, list):
        for image in request.image_url:
            content.append({"type": "image_url", "image_url": {"url": image, "detail": "high"}})
    else:
        content.append({"type": "image_url", "image_url": {"url": request.image_url, "detail": "high"}})

    messages = [
        {"role": "system", "content": extract_credit_data_from_pdf_prompt},
        {"role": "user", "content": content},
    ]
    llm = ChatOpenAI(
        model="gpt-5.2",
        reasoning_effort=request.reasoning_effort.value if request.reasoning_effort else "none",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(CreditDataExtractedFromPdf)
    extracted = await structured_llm.ainvoke(messages, config={"callbacks": [tracker]})
    if request.user_id:
        persist_pdf_extract_to_vector_store(request.user_id, extracted, request.file_id, tracker=tracker)
    _set_usage_headers(response, tracker)
    return extracted


class ErrorDispute(BaseModel):
    reason: str  = Field(description="Rason por la q el usuario quiere disputar");
    error: Union[ErrorTypeEnum, str] = Field(description="El error en cuestion, si es un error de Collection/Charge off/Repossession, poner Collection, Charge off o Repossession solamente, si no poner una descripcion del error");
    account_number: Optional[str]  = Field(description="El numero de cuenta asociado en caso de ser una cuenta");
    name_account: Optional[str] = Field(description="El nombre de cuenta o acredor asociado en caso de ser una cuenta");
    name_inquiry: Optional[str] = Field(description="El nombre del inquiry asociado en caso de ser un inquiry");
    credit_repo: str | list[str] = Field(description="El o los buros de credito implicados");
    inquiry_id: Optional[str] = Field(description="El identificador del inquiry en caso de ser un inquiry");
    inquiry_date: Optional[str] = Field(description="La fecha de solicitud del inquiry en caso de ser un inquiry, en formato yyyy-mm-dd");
    action: str = Field(description="La accion a tomar por el usuario(siempre va a ser para remover del reporte)");
    creditor: Optional[str] = Field(default=None, description="Acreedor de la cuenta en caso de ser una cuenta")
    creditor_data: Optional[Address] = Field(default=None, description="La direccion del acreedor y su nombre en caso de ser una cuenta")

class ErrorsDispute(BaseModel):
    errors: list[ErrorDispute]

class GetDisputesRequest(BaseModel):
    API_KEY: str
    user_id: str
    reasoning_effort: ReasoningEffortEnum = Field(default=ReasoningEffortEnum.NONE, description="El nivel de razonamiento a usar")

def get_clean_report(report: str):
    pattern_account_id = r'ID de la cuenta:\s*[a-fA-F0-9]{32}\.'

    report = re.sub(pattern_account_id, '', report).strip()

    report = re.sub(r"(Fecha de apertura de la cuenta:\s*\d{4}-\d{2}-\d{2}\.\s*)|(Fecha de ultima actividad:\s*\d{4}-\d{2}-\d{2}\.\s*)", "", report)

    report = re.sub(r"Mi primer nombre es:\s*(\w+)\nMi segundo nombre es:\s*(\w+)\nMi apellido es:\s*(\w+).*?(\d{4}-\d{2}-\d{2})", 
                r"Nombre: \1 \2 \3\nNacimiento: \4", report, flags=re.S)

    report = re.sub(r"Responsabilidad:\s*(.*?)\.", "", report)

    reemplazos = {
        " Tipo de Consulta: HARD;": "",
        "Pagos atrasados: 0.0": "Sin pagos atrasados",
        "Queda el: 0.0% para pagar de este prestamo": "Pagado por completo",
        "Buro de Credito": "Buro",
        "Estado de la cuenta": "Estado",
        "Nombre del acreedor:": "Acreedor:",
        "Tipo de fuente de plazo:": "Tipo/Fuente:",
        "Tipo de Fuente de Plazo:": "Tipo/Fuente:",
        "Segun lo acordado": "OK",
        "Ciudad ": "",
        "Estado ": "",
        "Codigo Postal ": "",
        "Calle ": "",
        "Cantidad de Pago Mensual: 0": "Sin Pago Mensual",
        "Cantidad de Pago Mensual:": "Pago mensual:",
        "Limite crediticio:": "LimiteCr:",
        "Pagado por completo. Sin pagos atrasados. Sin Pago Mensual. Estado: OK.": "Pagado por completo. Sin atrasos.",
        "para pagar de este prestamo": "por pagar",
        "Pagado por completo. LimiteCr: 0. Sin pagos atrasados. Sin Pago Mensual. Estado: OK.": "Pagado por completo. Sin atrasos. LimiteCr: 0.",
        "Numero de cuenta:": "Num. cuenta:",
        "Tipo/Fuente: Provided.": "",
    }
    for k, v in reemplazos.items():
        report = report.replace(k, v)

    # Redondear decimales a 2 decimales
    def round_decimals(match):
        num = float(match.group())
        return f"{num:.2f}"  # redondea a 2 decimales

    report = re.sub(r'(\d+\.\d+)', round_decimals, report)

    return report

@lru_cache(maxsize=100)
def get_clean_report_cached(report_str: str):
    """Versión cacheada para evitar procesamiento repetitivo de regex sobre el mismo reporte."""
    return get_clean_report(report_str)

def get_user_report(user_id:str, split: bool = False):
    collection = client.get_collection(name=get_collection_name(user_id))

    results = collection.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"source": {"$ne": "CreditSummary"}},
                {"source": {"$ne": "CreditScore"}},
                {"source": {"$ne": "Personal Info"}},
                {"field": {"$ne": "SSN"}},
                {"field": {"$ne": "credit_cards"}},
                {"field": {"$ne": "auto_loans"}},
                {"field": {"$ne": "education_loans"}},
                {"field": {"$ne": "mortgage_loans"}}
            ]
        },  # filter by user_id tag/metadata
        limit=None  # or a very high number if None is not supported
    )

    # Orden determinista: Chroma no garantiza el orden de .get(), y al re-indexar
    # (uuids nuevos) ese orden cambia, lo que rompía el hash del reporte y por tanto
    # el caché de get-disputes/verify-errors. Ordenar deja el reporte estable.
    disputes = sorted(results['documents'])

    if split:
        report_inquiries = "\n".join([dispute for dispute in disputes if 'Consulta' in dispute])
        # add creditor and account name to the report_inquiries
        report_inquiries = report_inquiries + "\n" + "Cuentas involucradas: "

        # get the creditor name from "Nombre del acreedor: <name>" to "." and is closed indicator
        account_creditor_names = ", ".join([
            re.search(r"Nombre del acreedor:\s*(.*?)\.", dispute).group(1)
            + (" (" + re.search(r"La cuenta esta\s*(.*?)\.", dispute).group(1) + "). " if re.search(r"La cuenta esta\s*(.*?)\.", dispute) else " (abierta). ")
            for dispute in disputes if 'Consulta' not in dispute
        ])
        report_inquiries = report_inquiries + "\n" + account_creditor_names

        report_accounts = "\n".join([dispute for dispute in disputes if 'Consulta' not in dispute])
        
        # report_accounts = []
        # for dispute in [dispute for dispute in disputes if 'Consulta' not in dispute]:
        #     number_account = re.search(r"Numero de cuenta:\s*(.*?)\.", dispute).group(1).replace("X", "")
        #     # if number_account is not on report_accounts, add it
        #     iidx_on_report_accounts = -1
        #     for idx, report_account in enumerate(report_accounts):
        #         number_account2 = re.search(r"Numero de cuenta:\s*(.*?)\.", report_account).group(1).replace("X", "")
        #         if number_account == number_account2:
        #             iidx_on_report_accounts = idx
        #             break
        #     if iidx_on_report_accounts == -1:
        #         report_accounts.append(dispute)
        #     else:
        #         report_accounts[iidx_on_report_accounts] = report_accounts[iidx_on_report_accounts] + "\n" + dispute
        
        # report_accounts = [get_clean_report(report_account) for report_account in report_accounts]

        return get_clean_report_cached(report_inquiries), get_clean_report_cached(report_accounts)

    report = "\n".join([dispute for dispute in disputes])
    report = get_clean_report_cached(report)

    return report


def normalize_repos_to_set(data):
    if data is None:
        return set()
    if isinstance(data, str):
        # Si es string, lo metemos en un set directamente
        return {data} 
    # Si es lista, la convertimos a set
    return set(data)

DISPUTE_FILTER_PHRASES = (
    "aparece con variaciones de nombre/datos",
    "se reporta con variación de nombre",
    "se reporta con nombres distintos",
    "aparece con variaciones de nombre",
    "aparece con variación de nombre",
    "se reporta con nombres/acreedor diferentes",
)

def _normalize_text(value: Optional[Union[str, ErrorTypeEnum]]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

def _should_filter_dispute(error: ErrorDispute) -> bool:
    if _looks_like_multiple_creditors(error.creditor) or _looks_like_multiple_creditors(error.name_account):
        return True
    text_to_scan = " ".join(
        [
            _normalize_text(error.reason),
        ]
    )
    return any(phrase in text_to_scan for phrase in DISPUTE_FILTER_PHRASES)

@app.post("/get-disputes")
async def get_disputes(request:GetDisputesRequest, response: Response) -> list[ErrorDispute]:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    report_inquiries, report_accounts = get_user_report(request.user_id, split=True)

    logger.error(f"Report inquiries: {report_inquiries}")
    logger.error(f"Report accounts: {report_accounts}")

    cache_key = _input_hash(report_inquiries, report_accounts, request.reasoning_effort)
    cached = _disputes_cache.get((request.user_id, cache_key))
    if cached is not None:
        _set_usage_headers(response, tracker)  # 0 tokens: servido desde caché
        return cached

    prompt_inquiries = f"""
    Eres un sistema de reparación de crédito y tu tarea es analizar los informes de los burós de crédito (Equifax, Experian y TransUnion) y detectar posibles errores relacionados únicamente con las inquiries.

    Acciones a realizar:
    1. Inquiries que no corresponden a cuentas abiertas:
        - Si una inquiry no corresponde a una cuenta abierta, se debe disputar para corregir esta incongruencia.
        - Los nombres de las cuentas asociadas a las inquiries no tienen que ser exactamente iguales.
        - No puedes disputar las inquiries que corresponden a cuentas abiertas, aunque no sean exactamente iguales los nombres de las cuentas asociadas a las inquiries.
        - No disputes ningun otro error que no sea una inquiry que no corresponda a una cuenta abierta.
    2. Manejo de Inquiries:
        - Disputar si NO CORRESPONDEN a una cuenta abierta o si la cuenta asociada está cerrada.

    Devuelve un JSON con un array errors donde cada objeto dentro del array tenga:
    - Rason por la q el usuario quiere disputar(reason)
    - El error en cuestion(error)
    - El nombre del inquiry asociado si es un inquiry(name_inquiry)
    - La fecha de solicitud del inquiry si es un inquiry, en formato yyyy-mm-dd(inquiry_date)
    - El o los buros de credito implicados, si el mismo error es en varios buros poner el error solo una vez, y decir los buros en los que esta, si es un error de un solo buro q tiene datos distintos(negativos) de los otros buros poner el error solo una vez y decir el buro en el que esta diferente, en formato de lista(credit_repo)
    - El identificador del inquiry si es un inquiry(inquiry_id)
    - La accion a tomar por el usuario(action)
    """

    prompt_accounts = f"""
    Eres un sistema de reparación de crédito y tu tarea es analizar los informes de los burós de crédito (Equifax, Experian, y TransUnion) y detectar posibles errores en las colecciones y otros elementos reportados para removerlos del reporte. A continuación, se detallan las acciones que debes realizar para identificar problemas comunes en los reportes de crédito y disputarlos si es necesario:
    1. Comparación de colecciones en los tres burós:
        - Compara la información de las colecciones reportadas por los tres burós.
        - No compara el nombre de la cuenta ni la direccion, estos pueden ser diferentes, solo el saldo y el estado.
        - Verifica que los saldos y los estados sean idénticos. Si no es así, genera una disputa.
    2. Estado de la colección:
        - Colección abierta incorrectamente: Una colección no debe estar en estado abierto si ya fue saldada o gestionada. Si se encuentra en estado abierto erróneamente, genera una disputa.
    3. Colección y cuenta original abiertas simultáneamente:
        - Si una cuenta original está abierta y tiene una colección asociada abierta, se debe disputar para corregir esta incongruencia. Ambas no deberían estar abiertas al mismo tiempo.
    4. Manejo de Marcas Negativas:
        - Late Payments: Enfocarse solo en el historial de pago tarde, no en la cuenta completa, incluso si la cuenta está pagada/cerrada.
        - Collection/Charge off/Repossession: Disputar la CUENTA COMPLETA para intentar su eliminación total. Si se identifica, la acción debe ser 'Disputar cuenta completa para eliminación'.

    Regla obligatoria: cada objeto de error corresponde a EXACTAMENTE UNA cuenta / un acreedor. NUNCA combines varios acreedores o cuentas en un mismo objeto; si un error afecta a varias cuentas, genera un objeto SEPARADO por cada una. `name_account` y `creditor` deben tener el nombre de UN SOLO acreedor tal cual aparece en el reporte.

    Devuelve un JSON con un array errors donde cada objeto dentro del array tenga:
    - Rason por la q el usuario quiere disputar(reason)
    - El error en cuestion, si es un error de Collection/Charge off/Repossession, poner Collection, Charge off o Repossession solamente(error)
    - El numero de cuenta asociado(account_number)
    - El nombre de cuenta asociado o el acreedor exacto como aparece en el reporte, nunca el tipo de cuenta(name_account)
    - El o los buros de credito implicados, si el mismo error es en varios buros poner el error solo una vez, y decir los buros en los que esta, si es un error de un solo buro q tiene datos distintos(negativos) de los otros buros poner el error solo una vez y decir el buro en el que esta diferente, en formato de lista(credit_repo)
    - La accion a tomar por el usuario(action)
    - Acreedor de la cuenta, el nombre exacto como aparece en el reporte(creditor)
    """
    messages_inquiries = [
        {"role": "system", "content": prompt_inquiries},
        {"role": "user", "content": f"Los informes de los tres burós se encuentran a continuación: {report_inquiries}"}
    ]
    llm = ChatOpenAI(
        model="gpt-5.2",
        reasoning_effort=request.reasoning_effort if request.reasoning_effort else "none",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ErrorsDispute)

    messages_accounts = [
        {"role": "system", "content": prompt_accounts},
        {"role": "user", "content": f"Los informes de los tres burós se encuentran a continuación: {report_accounts}"}
    ]

    cfg = {"callbacks": [tracker]}
    tasks = [structured_llm.ainvoke(messages_inquiries, config=cfg), structured_llm.ainvoke(messages_accounts, config=cfg)]

    llm_responses = await asyncio.gather(*tasks)

    errors = []

    for llm_response in llm_responses:
        errors.extend(llm_response.errors)

    errors = [error for error in errors if not _should_filter_dispute(error)]

    _cache_put(_disputes_cache, request.user_id, cache_key, errors)
    _set_usage_headers(response, tracker)
    return errors

class GetDisputesRequest(BaseModel):
    API_KEY: str
    image_url: Union[str, list[str]]
    reasoning_effort: ReasoningEffortEnum = Field(default=ReasoningEffortEnum.NONE, description="El nivel de razonamiento a usar")

@app.post("/get-disputes-by-pdf")
async def get_disputes_by_pdf(request:GetDisputesRequest, response: Response) -> list[ErrorDispute]:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    content = []

    if type(request.image_url) == list:
        for image in request.image_url:
            content.append({ 
            'type': 'image_url', 
            'image_url': { 'url': image, 'detail': 'auto'} 
        })
    else: 
        content.append({ 
            'type': 'image_url', 
            'image_url': { 'url': request.image_url, 'detail': 'auto'} 
        })
    
    messages = [
        {"role": "system", "content": get_disputes_by_pdf_prompt},
        {"role": "user", "content": content}
    ]
    llm = ChatOpenAI(
        model="gpt-5.2",
        reasoning_effort=request.reasoning_effort if request.reasoning_effort else "none",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ErrorsDispute)

    llm_response = await structured_llm.ainvoke(messages, config={"callbacks": [tracker]})

    _set_usage_headers(response, tracker)
    return llm_response.errors

import re
from datetime import date
from utils.letter_templates import first_round_template, second_round_template, third_round_template

class PersonalInfo(BaseModel):
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    address: str
    city: str
    state: str
    postal_code: str

class GenerateLetterRequest(BaseModel):
    API_KEY: str
    user_id: str
    sender: Optional[PersonalInfo] = None
    round: int
    errors: list[ErrorDispute]

class Letter(BaseModel):
    repo: str
    letter: str

class LetterCreditor(BaseModel):
    creditor: str = Field(description="The name of the creditor")
    letter: str = Field(description="The letter to be sent to the creditor")
    to: Address = Field(description="The address of the creditor and the company name")

class GenerateLetterResponse(BaseModel):
    letters: list[Letter]
    letters_creditor: list[LetterCreditor]
    sender: PersonalInfo

from utils.get_credit_repo_data import get_credit_repo_data

async def get_letter_content(llm, error, request, header, footer, curr_date, config=None):
    repo_data = get_credit_repo_data(error['repo'])

    prompt = f"""You are a letter-writing assistant. Given the user's personal information and a list of credit report errors, produce a formal dispute letter:
    Write the body of a dispute letter to TransUnion for the {request.round}th round of disputes. 
    Do NOT include any header, footer, contact information, dates, or signatures. 
    Only output the body text of the letter.
    The tone must escalate with each round, so third round must be the most aggressive, the second round must be more aggressive than the first round, and the first round must be the most polite.
    The letter should be written on english, its very important that the letter is written in english the content and every dispute item must be written in english.
    The letter should be written in a professional tone.
    The letter is for {error["repo"]} bureau, but dont introduce the letter like Dear bereau or anything like that , its just for you know the context.
    Do not mention any bureau unless it is necessary to reference data in the errors themselves
    However, you ARE allowed to reference any credit bureaus that appear inside the provided Errors data, but remember, the letter is for {error["repo"]} and the letter must NOT request or demand any actions from credit bureaus other than {error["repo"]}. 
    If the errors mention Experian, Equifax, or TransUnion, include those names exactly as they appear.
    
    Do not output anything except the completed letter text. Use the following input data:
    Errors: {error['errors']}"""

    response = await llm.ainvoke(prompt, config=config)

    return {
        'repo': error['repo'],
        'letter': f'{header}\n{error["repo"]}\n{repo_data["address"]}\n{repo_data["city"]}, {repo_data["state"]}, {repo_data["zip_code"]}\n\nDate: {curr_date}\n\nDear {error["repo"]},\n\n{response.content}\n{footer}'
    }

async def get_creditor_information(creditor, error, config=None):
    llm = ChatOpenAI(model="gpt-5.2", reasoning_effort="high")
    prompt = f"""You are a helpful research assistant. Use web search to find accurate, up-to-date information. You are given a creditor name and you need to find the mailing address for send a dispute letter information about the creditor on the US. This is i want to dispute:
    {error}
    Creditor: {creditor}"""
    structured_llm = llm.with_structured_output(Address)
    final_structured = await structured_llm.ainvoke(prompt, config=config)
    return final_structured

async def get_letter_content_creditor(llm, errors, creditor, creditor_data, request, header, footer, curr_date, config=None) -> LetterCreditor:
    prompt = f"""You are a letter-writing assistant. Given the user's personal information and a list of credit report errors, produce a formal dispute letter:
    Write the body of a dispute letter to {creditor} for the {request.round}th round of disputes. 
    Do NOT include any header, footer, contact information, dates, or signatures. 
    Only output the body text of the letter.
    The tone must escalate with each round, so third round must be the most aggressive, the second round must be more aggressive than the first round, and the first round must be the most polite.
    The letter should be written on english, its very important that the letter is written in english the content and every dispute item must be written in english.
    
    The letter is for {creditor}, but dont introduce the letter like Dear "Creditor Name" or anything like that , its just for you know the context.
    Do not mention any bureau unless it is necessary to reference data in the errors themselves    
    Do not output anything except the completed letter text. Use the following input data:
    Errors: {errors}"""

    response = await llm.ainvoke(prompt, config=config)

    creditor_information = None

    if creditor_data is not None:
        creditor_information = creditor_data
    else:
        creditor_information = await get_creditor_information(creditor, errors, config=config)

    creditor_name = creditor or creditor_information.company_name

    return {
        'creditor': creditor_name,
        'to': creditor_information,
        'letter': f'{header}\n{creditor_name}\n{creditor_information.address}\n{creditor_information.city}, {creditor_information.state}, {creditor_information.zip_code}\n\nDate: {curr_date}\n\nDear {creditor_name},\n\n{response.content}\n{footer}'
    }

@app.post("/generate-letter")
async def generate_letter(request:GenerateLetterRequest, response: Response) -> GenerateLetterResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()
    cfg = {"callbacks": [tracker]}

    # Idempotencia: la misma solicitud (mismos errores, round y sender) devuelve
    # las mismas cartas sin volver a llamar al LLM. Protege contra reintentos que
    # de otro modo regenerarían cartas y volverían a gastar tokens.
    idem_key = _input_hash(request.model_dump_json(exclude={"API_KEY"}))
    cached = _letters_cache.get((request.user_id, idem_key))
    if cached is not None:
        _set_usage_headers(response, tracker)  # 0 tokens: idempotente
        return cached

    collection = client.get_collection(name=get_collection_name(request.user_id))

    results = collection.get(
        where={
            "$and": [
                {"user_id": request.user_id},
                {"source": "Personal Info"}
            ]
        },  # filter by user_id tag/metadata
        limit=None  # or a very high number if None is not supported
    )

    personal_info = results['documents']

    # Extraer nombre completo
    first_name = ""
    middle_name = ""
    last_name = ""
    # Extraer dirección actual
    address = ""
    city = ""
    state = ""
    postal_code = ""

    if request.sender:
        first_name = request.sender.first_name
        
        if request.sender.middle_name:
            middle_name = request.sender.middle_name
        
        last_name = request.sender.last_name
        address = request.sender.address
        city = request.sender.city
        state = request.sender.state
        postal_code = request.sender.postal_code 
    else:
        for line in personal_info:
            if "primer nombre" in line:
                first_name = re.search(r": (.+)$", line).group(1)
            elif "segundo nombre" in line:
                match = re.search(r": (.+)$", line)
                if match:
                    middle_name = match.group(1)
            elif "apellido" in line:
                last_name = re.search(r": (.+)$", line).group(1)

        for line in personal_info:
            if "Residiendo Actualmente" in line:
                city = re.search(r"Ciudad ([^,;]+)", line).group(1)
                state = re.search(r"Estado (\w{2})", line).group(1)
                postal_code = re.search(r"Codigo Postal (\d+)", line).group(1)
                address = re.search(r"Calle ([^;]+)", line).group(1)
                break
    
    # other info for letter
    bdate = ""
    ssn = ""

    for line in personal_info:
        if "Mi Fecha de nacimiento es: " in line:
            bdate = re.search(r": (.+)$", line).group(1)
        if "Mi SSN es: " in line:
            ssn = re.search(r": (.+)$", line).group(1)

    full_name = " ".join([first_name, middle_name, last_name]).strip()

    # Obtener fecha actual
    curr_date = date.today().isoformat()  # 'YYYY-MM-DD'

    header = "\n".join([full_name, address, f"{city}, {state} {postal_code}"]) + "\n"

    # DOB: [bdate]    SSN: [ss_number]
    header += f"\nDOB: {bdate}    SSN: {ssn}\n\n"

    footer = f"\nSincerely,\n\n{full_name}"

    llm = ChatOpenAI(model="gpt-5.2", reasoning_effort="low")

    equifax_errors = [
        {
            'reason': error.reason,
            'error': error.error,
            'account_number': error.account_number,
            'name_account': error.name_account,
            'name_inquiry': error.name_inquiry,
            'inquiry_id': error.inquiry_id,
            'inquiry_date': error.inquiry_date,
            'action': error.action,
            'creditor': error.creditor, 
            'creditor_data': error.creditor_data
        } for error in request.errors
        if (
            (isinstance(error.credit_repo, list) and "Equifax" in error.credit_repo)
            or (isinstance(error.credit_repo, str) and "Equifax" in error.credit_repo)
        )
    ]

    experian_errors = [
        {
            'reason': error.reason,
            'error': error.error,
            'account_number': error.account_number,
            'name_account': error.name_account,
            'name_inquiry': error.name_inquiry,
            'inquiry_id': error.inquiry_id,
            'inquiry_date': error.inquiry_date,
            'action': error.action,
            'creditor': error.creditor, 
            'creditor_data': error.creditor_data
        } for error in request.errors
        if (
            (isinstance(error.credit_repo, list) and "Experian" in error.credit_repo)
            or (isinstance(error.credit_repo, str) and "Experian" in error.credit_repo)
        )
    ]
    transunion_errors = [
        {
            'reason': error.reason,
            'error': error.error,
            'account_number': error.account_number,
            'name_account': error.name_account,
            'name_inquiry': error.name_inquiry,
            'inquiry_id': error.inquiry_id,
            'inquiry_date': error.inquiry_date,
            'action': error.action,
            'creditor': error.creditor, 
            'creditor_data': error.creditor_data
        } for error in request.errors
        if (
            (isinstance(error.credit_repo, list) and "TransUnion" in error.credit_repo)
            or (isinstance(error.credit_repo, str) and "TransUnion" in error.credit_repo)
        )
    ]

    error_list = [
        {
            'repo': 'Equifax',
            'errors': equifax_errors
        },
        {
            'repo': 'Experian',
            'errors': experian_errors
        },
        {
            'repo': 'TransUnion',
            'errors': transunion_errors
        }
    ]

    letters_generated = []
    tasks = []

    # get charge-of and collections errors

    for error in error_list:
        if len(error['errors']):
            tasks.append(
                get_letter_content(llm, error, request, header, footer, curr_date, config=cfg)
            )

    if request.round > 3:
        error_list_creditor = {}
    
        for error in error_list:
            if len(error['errors']) > 0:
                for error_item in error['errors']:
                    if ('creditor' in error_item) and (error_item['error'] in [ErrorTypeEnum.COLLECTION, ErrorTypeEnum.CHARGE_OFF]):
                        if error_item['creditor'] not in error_list_creditor:
                            error_list_creditor[error_item['creditor']] = []
                        error_list_creditor[error_item['creditor']].append(error_item)

        for creditor, errors in error_list_creditor.items():
            creditor = None
            creditor_data = None
            for error_item in errors:
                if creditor is None:
                    creditor = error_item['creditor']
                if creditor_data is None:
                    creditor_data = error_item['creditor_data']
                
                if creditor and creditor_data:
                    break

            tasks.append(
                get_letter_content_creditor(
                    llm,
                    errors,
                    creditor,
                    creditor_data,
                    request,
                    header,
                    footer,
                    curr_date,
                    config=cfg
                )
            )


    letters_generated = await asyncio.gather(*tasks)
    letters = []
    letters_creditor = []

    for letter in letters_generated:
        if 'repo' in letter:
            letters.append(letter)
        elif 'creditor' in letter:
            letters_creditor.append(letter)

    result = {
        'letters': letters,
        'letters_creditor': letters_creditor,
        'sender': {
            'first_name': first_name,
            'middle_name': middle_name,
            'last_name': last_name,
            'address': address,
            'city': city,
            'state': state,
            'postal_code': postal_code
        }
    }

    _cache_put(_letters_cache, request.user_id, idem_key, result)
    _set_usage_headers(response, tracker)
    return result

class ErrorDisputeWithId(ErrorDispute):
    id: str

class VerifyErrorsRequest(BaseModel):
    errors: list[ErrorDisputeWithId]
    API_KEY: str
    user_id: str

class RepoError(BaseModel):
    id: str = Field(description="El identificador del error")
    TransUnion: bool = Field(description="Si esta presente en TransUnion")
    Equifax: bool = Field(description="Si esta presente en Equifax")
    Experian: bool = Field(description="Si esta presente en Experian")

class VerifyErrorsResponse(BaseModel):
    still_on_report: list[RepoError] = Field(description="Si esta o no el error, separado por reporte");

def get_error_string(error: Optional[ErrorDisputeWithId] | ErrorDispute) -> str:
    error_string = ""
    if isinstance(error, ErrorDisputeWithId) and isinstance(error.id, str):
        error_string += f"Identificador del error: {error.id}\n"
    if isinstance(error.name_account, str):
        error_string += f"Nombre de la cuenta: {error.name_account}\n"
    if isinstance(error.account_number, str):
        error_string += f"Numero de la cuenta: {error.account_number}\n"
    if isinstance(error.name_inquiry, str):
        error_string += f"Nombre del inquiry: {error.name_inquiry}\n"
    if isinstance(error.inquiry_id, str):
        error_string += f"Identificador del inquiry: {error.name_inquiry}\n"

    if isinstance(error.credit_repo, str):
        error_string += f"Buro de credito: {error.credit_repo}\n"
    else:
        error_string += f"Buros de credito: {", ".join([repo for repo in error.credit_repo])}\n"

    if isinstance(error.reason, str):
        error_string += f"Rason de la disputa: {error.reason}"
    if isinstance(error.error, str):
        error_string += f"Error en cuestion: {error.error}"

    return error_string

BATCH_SIZE_VERIFY_ERRORS = 10

def _verify_errors_batch(errors_batch: list, report: str, config: dict = None) -> VerifyErrorsResponse:
    """Verifica un lote de hasta 10 errores contra el reporte. Uso interno en paralelo."""
    errors_text = "\n"
    for i, error in enumerate(errors_batch, 1):
        errors_text += f"{i}- "
        errors_text += get_error_string(error)
        errors_text += "\n"
    prompt = f"""
        Eres un sistema de verificacion de errores en el credito, debes devolver de manera ordenada si estan aun presentes o no en el credito los siguientes errores, analizalos uno por uno, ten todo en cuenta, y responde por cada error Verdadero(si esta presente el error en el reporte de credito), Falso(si no aparece en el credito), esto por cada buro de credito (Experian, TransUnion, Equifax), junto con el identificador del Error:
        
        Errores:
        {errors_text}
        Los informes de los tres burós se encuentran a continuación:
        {report}
    """
    llm = ChatOpenAI(model="gpt-5.2", reasoning_effort=ReasoningEffortEnum.MEDIUM)
    structured_llm = llm.with_structured_output(VerifyErrorsResponse)
    return structured_llm.invoke(prompt, config=config)

@app.post("/verify-errors")
def verify_errors(request: VerifyErrorsRequest, response: Response) -> VerifyErrorsResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    report = get_user_report(request.user_id)
    errors = [error for error in request.errors if not _should_filter_dispute(error)]
    if not errors:
        _set_usage_headers(response, tracker)
        return VerifyErrorsResponse(still_on_report=[])

    # El resultado es determinista respecto al reporte + los errores a verificar
    cache_key = _input_hash(report, "".join(get_error_string(e) for e in errors))
    cached = _verify_cache.get((request.user_id, cache_key))
    if cached is not None:
        _set_usage_headers(response, tracker)  # 0 tokens: servido desde caché
        return cached

    # Partir en lotes de máximo 10
    batches = [errors[i : i + BATCH_SIZE_VERIFY_ERRORS] for i in range(0, len(errors), BATCH_SIZE_VERIFY_ERRORS)]
    all_still_on_report: list[RepoError] = []

    cfg = {"callbacks": [tracker]}
    # Procesar lotes en paralelo, con un tope de hilos para no disparar decenas
    # de llamadas simultáneas si llega una lista de errores muy grande.
    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        futures = {executor.submit(_verify_errors_batch, batch, report, cfg): batch for batch in batches}
        for future in as_completed(futures):
            batch_response = future.result()
            all_still_on_report.extend(batch_response.still_on_report)

    result = VerifyErrorsResponse(still_on_report=all_still_on_report)
    _cache_put(_verify_cache, request.user_id, cache_key, result)
    _set_usage_headers(response, tracker)
    return result

class CompareErrorsRequest(BaseModel):
    errors_1: list[ErrorDispute]
    errors_2: list[ErrorDisputeWithId]
    API_KEY: str

class CompareErrorsResponse(BaseModel):
    same_errors_ids: list[str]

@app.post("/compare-errors")
def compare_errors(request: CompareErrorsRequest, response: Response) -> CompareErrorsResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    errors_1_string = ""
    errors_2_string = ""

    for error in request.errors_1:
        errors_1_string += get_error_string(error)
        errors_1_string += "\n"
    for error in request.errors_2:
        errors_2_string += get_error_string(error)
        errors_2_string += "\n"

    prompt = f"""
        Eres un sistema de comparacion de errores en el credito, debes comparar los siguientes errores y devolver los identificadores de los errores que sean los mismos en ambos errores:
        Ten en cuenta que si los errores son de diferentes burós, no son los mismos errores.
        Si existen datos como fecha de apertura, fecha de ultima actividad, balance, etc, que sean diferentes, no son los mismos errores.
        Si el error es un inquiry aunque tenga el mismo nombre, si tiene diferente fecha de solicitud, no son los mismos errores.
        
        Errores 1:
        {errors_1_string}
        Errores 2:
        {errors_2_string}
        Devuelve un JSON con un array same_errors_ids con los identificadores de los errores que sean los mismos en ambos errores.
        Solo compara si estan en dos listados diferentes, si estan en el mismo listado repetidos no.
    """

    llm = ChatOpenAI(model="gpt-5-mini")
    structured_llm = llm.with_structured_output(CompareErrorsResponse)
    result = structured_llm.invoke(prompt, config={"callbacks": [tracker]})
    _set_usage_headers(response, tracker)
    return result

class LitigationErrorTypeEnum(str, Enum):
    POST_BK_NO_DISCLOSURE = "Reporte posterior a bancarrota sin divulgacion de bancarrota"
    DOUBLE_REPORTING = "Doble reporte de la misma cuenta"
    MIXED_FILE = "Archivo mezclado"
    MULTIPLE_SSN = "Multiples SSN con cuentas que no son del consumidor"
    DOWNSTREAM_DEBT_BUYER_MISMATCH = "Comprador de deuda reporta informacion distinta al acreedor original"
    POST_OBSOLESCENCE = "Reporte posterior a la fecha de obsolescencia"
    POST_DISPUTE_NO_NOTATION = "Reporte posterior a disputa sin notacion de disputa"
    DECEASED_WHILE_LIVING = "Consumidor reportado como fallecido estando vivo"

class LitigationError(BaseModel):
    error_type: LitigationErrorTypeEnum = Field(description="El tipo de error litigable, uno de los ocho valores del enum")
    reason: str = Field(description="Razon / descripcion de por que es un error litigable")
    evidence: str = Field(description="La evidencia textual o datos del informe que sustentan el error")
    name_account: Optional[str] = Field(default=None, description="Nombre de UNA sola cuenta o acreedor, exacto como aparece en el reporte. Nunca combines varios acreedores aqui")
    account_number: Optional[str] = Field(default=None, description="Numero de cuenta asociado (de una sola cuenta), si aplica")
    creditor: Optional[str] = Field(default=None, description="Acreedor de UNA sola cuenta, el nombre exacto como aparece en el reporte. Nunca combines varios acreedores aqui")
    credit_repo: Union[str, list[str]] = Field(description="El o los buros de credito donde aparece esta misma cuenta (Equifax/Experian/TransUnion)")

class LitigationErrors(BaseModel):
    errors: list[LitigationError]

class GetLitigationErrorsRequest(BaseModel):
    API_KEY: str
    user_id: str
    reasoning_effort: ReasoningEffortEnum = Field(default=ReasoningEffortEnum.HIGH, description="El nivel de razonamiento a usar")

def _looks_like_multiple_creditors(value: Optional[str]) -> bool:
    """Un error de litigacion es de UNA sola cuenta. Si el acreedor/nombre viene
    concatenado (varios acreedores en un mismo campo) es basura de la IA que se debe
    descartar. El punto y coma casi nunca aparece en el nombre real de un acreedor,
    asi que es una senal de alta precision de que la IA agrupo varias cuentas."""
    if not value:
        return False
    return ";" in value

def _should_filter_litigation(error: "LitigationError") -> bool:
    return _looks_like_multiple_creditors(error.creditor) or _looks_like_multiple_creditors(error.name_account)

def get_user_litigation_report(user_id: str) -> str:
    """Reporte completo para analisis de litigacion: incluye informacion personal,
    registros publicos y cuentas (con fechas y SSN intactos), excluyendo solo los
    resumenes/puntajes y los agregados que no aportan a la deteccion legal."""
    collection = client.get_collection(name=get_collection_name(user_id))

    results = collection.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"source": {"$ne": "CreditSummary"}},
                {"source": {"$ne": "CreditScore"}},
                {"source": {"$ne": "General Knowledge"}},
                {"field": {"$ne": "credit_cards"}},
                {"field": {"$ne": "auto_loans"}},
                {"field": {"$ne": "education_loans"}},
                {"field": {"$ne": "mortgage_loans"}},
            ]
        },
        limit=None,
    )

    # Orden determinista para que el hash del reporte (y el caché) sea estable
    # aunque se re-indexe y cambien los uuids/orden de Chroma.
    return "\n".join(sorted(results['documents']))

@app.post("/get-litigation-errors")
async def get_litigation_errors(request: GetLitigationErrorsRequest, response: Response) -> list[LitigationError]:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    report = get_user_litigation_report(request.user_id)

    cache_key = _input_hash(report, request.reasoning_effort, get_litigation_errors_prompt)
    cached = _litigation_cache.get((request.user_id, cache_key))
    if cached is not None:
        _set_usage_headers(response, tracker)  # 0 tokens: servido desde caché
        return cached

    messages = [
        {"role": "system", "content": get_litigation_errors_prompt},
        {"role": "user", "content": f"Los informes de los tres burós se encuentran a continuación: {report}"}
    ]
    llm = ChatOpenAI(
        model="gpt-5.2",
        reasoning_effort=request.reasoning_effort if request.reasoning_effort else ReasoningEffortEnum.HIGH,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(LitigationErrors)

    llm_response = await structured_llm.ainvoke(messages, config={"callbacks": [tracker]})

    errors = [e for e in llm_response.errors if not _should_filter_litigation(e)]

    _cache_put(_litigation_cache, request.user_id, cache_key, errors)
    _set_usage_headers(response, tracker)
    return errors


# %% Lessons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@app.post("/add-lesson")
async def add_lesson(request: AddLessonRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    lesson = request.lesson
    def upsert():
        # Dummy embedding: lessons are retrieved with get(), not vector search
        get_lessons_collection().upsert(
            ids=[lesson.lesson_id],
            embeddings=[[0.0]],
            metadatas=[{
                "lesson_id": lesson.lesson_id,
                "title": lesson.title,
                "description": lesson.description,
                "level_hint": lesson.level_hint,
            }],
        )
    await asyncio.to_thread(upsert)
    return {"ok": True}


@app.get("/get-lessons")
async def get_lessons(api_key: str = Query(alias="api_key")):
    if os.getenv("API_KEY") != api_key:
        raise HTTPException(status_code=400, detail="Api key dont match")

    def fetch():
        return get_lessons_collection().get()
    results = await asyncio.to_thread(fetch)

    lessons = []
    for meta in results.get("metadatas", []):
        if meta:
            lessons.append(Lesson(
                lesson_id=meta["lesson_id"],
                title=meta["title"],
                description=meta["description"],
                level_hint=meta["level_hint"],
            ))
    return lessons


# %% Credit Plan %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@app.post("/generate-plan")
async def generate_plan(request: GeneratePlanRequest, response: Response) -> CreditPlan:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    tracker = TokenUsageTracker()

    report_summary_text = await asyncio.to_thread(get_user_report, request.user_id)

    def fetch_lessons():
        return get_lessons_collection().get()
    lesson_results = await asyncio.to_thread(fetch_lessons)

    lessons_text = "\n".join([
        f"- ID: {m['lesson_id']} | Nivel sugerido: {m['level_hint']} | Título: {m['title']} | Descripción: {m['description']}"
        for m in lesson_results.get("metadatas", []) if m
    ]) or "No hay lecciones disponibles."

    # El plan se mantiene hasta que su propia duración (en meses) se cumpla; pasado
    # ese tiempo, o si el reporte cambia, se genera uno nuevo.
    cache_key = _input_hash(report_summary_text, lessons_text)
    cached = _plan_cache.get((request.user_id, cache_key))
    if cached is not None:
        cached_plan, expiry_ts = cached
        if time.time() < expiry_ts:
            _set_usage_headers(response, tracker)  # 0 tokens: servido desde caché
            return cached_plan

    prompt = f"""
Eres un experto en reparación de crédito en Estados Unidos. Basándote en el reporte de crédito del usuario,
genera un plan personalizado de 5 niveles con tareas semanales concretas y accionables.

NIVELES (usa exactamente estos nombres en el campo `name`):
1. Conoce tu crédito
2. Construyendo tu base
3. Optimizando
4. Ampliando
5. Crédito Pro

REGLAS ESTRICTAS:
- Cada nivel contiene 1 o 2 meses (MonthPlan). Cada mes tiene exactamente 4 tareas (una por semana, weeks 1 a 4).
- task_type debe ser exactamente uno de: "action", "dispute", "lesson".
    - "action": pago de deuda, apertura de cuenta asegurada, reducción de utilización, solicitar aumento de límite, etc.
    - "dispute": disputar un error específico del reporte (inquiry, colección, charge-off, etc.).
    - "lesson": completar una lección del curso. Usa el lesson_id exacto de la lista de lecciones.
- Si no hay lecciones disponibles o no aplica ninguna, no generes tareas de tipo "lesson".
- estimated_score_gain: puntos de crédito estimados que el usuario ganará ESE mes (entero positivo, realista según su perfil).
- Priorización dentro de cada mes: primero reducción de utilización, luego disputas, luego nuevos hábitos o lecciones.
- status: nivel 1 → "in_progress", niveles 2-5 → "locked".
- Los títulos de las tareas deben ser específicos al usuario (menciona acreedores reales, montos reales del reporte).

REPORTE DEL USUARIO:
{report_summary_text}

LECCIONES DISPONIBLES:
{lessons_text}
"""

    llm = ChatOpenAI(model="gpt-5.2", temperature=0)
    structured_llm = llm.with_structured_output(CreditPlan)
    plan = await structured_llm.ainvoke(prompt, config={"callbacks": [tracker]})

    # Duración total del plan = suma de meses de todos los niveles. El caché expira
    # al completarse esa duración, momento en que se generará otro plan.
    total_months = sum(len(level.months) for level in plan.levels) or 1
    expiry_ts = time.time() + total_months * 30 * 24 * 60 * 60
    _cache_put(_plan_cache, request.user_id, cache_key, (plan, expiry_ts))
    _set_usage_headers(response, tracker)
    return plan


# insert_general_knowledge()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
