import os
from dotenv import load_dotenv
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# imports
from fastapi import FastAPI, Request, HTTPException
import logging
from fastapi.responses import JSONResponse
from uuid import uuid4
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
from models import CreditRequest
from utils.get_score_rating import get_score_rating
from utils.prompts import scan_documents


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

credit_db_dir = "./credit_db"

# Helper function para obtener vector store (reutiliza conexiones)
def get_vector_store(user_id: str) -> Chroma:
    """
    Obtiene o crea una instancia de Chroma para un usuario específico.
    Reutiliza la conexión cuando es posible.
    """
    return Chroma(
        collection_name=f"{user_id}_credit_collection",
        embedding_function=embeddings,
        persist_directory=credit_db_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


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
            "path": request.url.path
        }
    )


# routes
@app.post("/add-user-credit-data")
async def add_user_credit_data(historic_credit:CreditRequest):
        if os.getenv("API_KEY") != historic_credit.API_KEY:
            raise HTTPException(status_code=400, detail="Api key dont match")
        documents = []

        # log the request
        logger.error(f"Adding user credit data for user", historic_credit)
         
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

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store = get_vector_store(historic_credit.USER_ID)
        response = vector_store.get(where={"user_id": historic_credit.USER_ID})
        if len(response['documents']) > 0:
            vector_store.delete(where={"user_id": historic_credit.USER_ID})

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

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

class QueryResponse(BaseModel):
    answer: str

# endpoint to retrieve an answer
@app.post("/query")
async def query(query_request:QueryRequest) -> QueryResponse:
    if os.getenv("API_KEY") != query_request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
    
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
    response = chain.invoke({"input": query_request.query})
    return {
        'answer': response['answer'],
    }

class AiAnswer(BaseModel):
    answer: str = Field(description="Respuesta");
    must_talk_with_a_human: bool = Field(description="Si el usuario debe contactar o no con un humano para la pregunta que esta haciendo");

class PosAiAnswer(BaseModel):
    must_talk_with_a_human: bool = Field(description="Si el usuario debe contactar o no con un humano para la pregunta que esta haciendo");

@app.post("/query-without-limits")
async def query_without_limits(query_request:QueryRequest) -> AiAnswer:
    if os.getenv("API_KEY") != query_request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

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
    response = chain.invoke({"input": query_request.query})

    llm = ChatOpenAI()

    structured_llm = llm.with_structured_output(PosAiAnswer)

    prompts = [
        SystemMessage("Identifica si es indispensable la intervencion de una persona humana en este contexto."),
    ]

    for m in query_request.last_messages:
        prompts.append(HumanMessage(content=m.input))
        prompts.append(AIMessage(content=m.output))

    prompts.append(HumanMessage(content=query_request.query))

    pos_response = structured_llm.invoke(prompts)
    
    return {
        'answer': response['answer'],
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

    vector_store = get_vector_store(request.user_id)
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

    vector_store = get_vector_store(request.user_id)
    vector_store.delete(where={"user_id": request.user_id})
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
async def scan_image(request: ScanImageRequest) -> DocumentData:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
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
    vision_response = structured_llm.invoke(prompts)
    return vision_response

# paraphrase letter in english
class ParaphraseLetterResponse(BaseModel):
    paraphrased_letter: str

class ParaphraseLetterRequest(BaseModel):
    API_KEY: str
    letter: str

@app.post("/paraphrase-letter")
async def paraphrase_letter(request: ParaphraseLetterRequest):
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")
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
    response = structured_llm.invoke(prompts)
    return response

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


class ErrorDispute(BaseModel):
    reason: str  = Field(description="Rason por la q el usuario quiere disputar");
    error: str  = Field(description="El error en cuestion");
    account_number: Optional[str]  = Field(description="El numero de cuenta asociado en caso de ser una cuenta");
    name_account: Optional[str] = Field(description="El nombre de cuenta asociado en caso de ser una cuenta");
    name_inquiry: Optional[str] = Field(description="El nombre del inquiry asociado en caso de ser un inquiry");
    credit_repo: str | list[str] = Field(description="El o los buros de credito implicados");
    inquiry_id: Optional[str] = Field(description="El identificador del inquiry en caso de ser un inquiry");
    inquiry_date: Optional[str] = Field(description="La fecha de solicitud del inquiry en caso de ser un inquiry, en formato yyyy-mm-dd");
    action: str = Field(description="La accion a tomar por el usuario(siempre va a ser para remover del reporte)");

class ErrorsDispute(BaseModel):
    errors: list[ErrorDispute]

class GetDisputesRequest(BaseModel):
    API_KEY: str
    user_id: str

def get_user_report(user_id:str):
    vector_store = get_vector_store(user_id)
    results = vector_store.get(
        where={
            "$and": [
                {"user_id": user_id},
                {"source": {"$ne": "CreditSummary"}},
                {"source": {"$ne": "CreditScore"}},
                {"field": {"$ne": "SSN"}},
                {"field": {"$ne": "credit_cards"}},
                {"field": {"$ne": "auto_loans"}},
                {"field": {"$ne": "education_loans"}},
                {"field": {"$ne": "mortgage_loans"}}
            ]
        },  # filter by user_id tag/metadata
        limit=None  # or a very high number if None is not supported
    )

    disputes = results['documents']

    pattern_account_id = r'ID de la cuenta:\s*[a-fA-F0-9]{32}\.'

    report = "\n".join([dispute for dispute in disputes])

    total_characters = len(report)

    report = re.sub(pattern_account_id, '', report).strip()

    report = re.sub(r"Mi primer nombre es:\s*(\w+)\nMi segundo nombre es:\s*(\w+)\nMi apellido es:\s*(\w+).*?(\d{4}-\d{2}-\d{2})", 
                r"Nombre: \1 \2 \3\nNacimiento: \4", report, flags=re.S)

    reemplazos = {
        " Tipo de Consulta: HARD;": "",
        "Fecha de apertura de la cuenta:": "Abierta el:",
        "Fecha de ultima actividad:": "Ultima actividad:",
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
        "Responsabilidad:": "Resp.",
        "Numero de cuenta:": "Num. cuenta:"
    }
    for k, v in reemplazos.items():
        report = report.replace(k, v)


    # Redondear decimales a 2 decimales
    def round_decimals(match):
        num = float(match.group())
        return f"{num:.2f}"  # redondea a 2 decimales

    report = re.sub(r'(\d+\.\d+)', round_decimals, report)

    total_characters_after = len(report)

    logger.error(f"Total de caracteres antes: {total_characters}, Total de caracteres despues: {total_characters_after}, Diferencia: {total_characters - total_characters_after}")

    return report

@app.post("/get-disputes")
def get_disputes(request:GetDisputesRequest) -> list[ErrorDispute]:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    report = get_user_report(request.user_id)

    prompt = f"""
    Eres un sistema de reparación de crédito y tu tarea es analizar los informes de los burós de crédito (Equifax, Experian, y TransUnion) y detectar posibles errores en las colecciones y otros elementos reportados para removerlos del reporte. A continuación, se detallan las acciones que debes realizar para identificar problemas comunes en los reportes de crédito y disputarlos si es necesario:
    1. **Comparación de colecciones en los tres burós:**
        - Compara la información de las colecciones reportadas por los tres burós.
        - Verifica que los saldos, las fechas y los estados sean idénticos. Si no es así, genera una disputa.
    2. **Verificación de información errónea:**
        - **Balance incorrecto:** Si el balance de la deuda registrado es erróneo, marca este dato para ser disputado.
        - **Fecha incorrecta:** Verifica que las fechas de la última actividad y la fecha de apertura sean correctas. Si alguna de estas fechas está equivocada, se debe disputar.
        - **Fecha de última actividad:** Esta fecha debe ser precisa. Si no lo es, disputa el dato.
    3. **Estado de la colección:**
        - **Colección abierta incorrectamente:** Una colección no debe estar en estado abierto si ya fue saldada o gestionada. Si se encuentra en estado abierto erróneamente, genera una disputa.
    4. **Colección duplicada:**
        - Si una misma colección está reportada en más de un buró, o si aparece duplicada dentro del mismo buró, se debe disputar la eliminación de la entrada duplicada.
    5. **Colección y cuenta original abiertas simultáneamente:**
        - Si una cuenta original está abierta y tiene una colección asociada abierta, se debe disputar para corregir esta incongruencia. Ambas no deberían estar abiertas al mismo tiempo.
    6. **Marcas negativas a buscar:**
    MARCAS NEGATIVAS:
    Late payments: cuando en una cuenta se han hecho pagos fuera de tiempo aparece historial de pago tarde, pueden ser por 30, 60, 90, 120, 150 o 180 días. Las marcas de pago tarde se quedan en el reporte aunque la deuda se pague e incluso si la cuenta se cierra. En el caso de estas marcas negativas se trabaja solo el historial de pago tarde, no la cuenta completa. 
    Collection/Charge off: cuando una cuenta llega a los 180 o más días de pago tarde, los creditores la marcan como collection o charge off, es decir ya no solo se debe la cantidad que está como “pago tarde” sino el balance completo de la cuenta. En esos casos las líneas de crédito se cierran. Las marcas de collection o charge off son las que más afectan al crédito y en estos casos se disputa la cuenta completa con el fin de que se elimine del todo sin que se tenga que pagar. En los casos donde no se eliminen las colecciones después de 3 rondas de disputa seguidas, se presenta al cliente la alternativa de buscar acuerdos de pago, para que esa cuenta aparezca con balance $0 y de esa forma deje de afectar el crédito. 
    Repossession: cuando un préstamo de carro no es pagado se marca como repossession, en cuestiones del crédito es parecido a una colección, con la diferencia de que el balance que se reporta en deuda es el que queda después de que el banco recupera el auto, lo subastan y lo que ganan en la subasta lo descuentan de la deuda total. Suelen ser un poco más complicadas de eliminar y de obtener acuerdos de pago. 
    Inquiries: marcas que dejan las revisiones que hacen los bancos antes de autorizar un préstamo o una línea de crédito. En la reparación solo se pueden trabajar los que correspondan a cuentas marcadas como cerradas o que no se hayan autorizado, es decir no aparezcan en el reporte de crédito, es recomendable esperar un mes para disputar un nuevo inquiry porque a veces pueden tardar una semanas
    Devuelve un JSON con un array errors donde cada objeto dentro del array tenga:
    - Rason por la q el usuario quiere disputar(reason)
    - El error en cuestion(error)
    - El numero de cuenta asociado en caso de ser una cuenta(account_number)
    - El nombre de cuenta asociado en caso de ser una cuenta(name_account)
    - El nombre del inquiry asociado en caso de ser un inquiry(name_inquiry)
    - La fecha de solicitud del inquiry en caso de ser un inquiry, en formato yyyy-mm-dd(inquiry_date)
    - El o los buros de credito implicados, si el mismo error es en varios buros poner el error solo una vez, y decir los buros en los que esta, en formato de lista(credit_repo)
    - El identificador del inquiry en caso de ser un inquiry(inquiry_id)
    - La accion a tomar por el usuario(action)
    Los informes de los tres burós se encuentran a continuación:
    
    {report}
    """
    llm = ChatOpenAI(model="gpt-5")
    structured_llm = llm.with_structured_output(ErrorsDispute)
    response = structured_llm.invoke(prompt)
    return response.errors

import re
from datetime import date
from utils.letter_templates import first_round_template, second_round_template, third_round_template

class GenerateLetterRequest(BaseModel):
    API_KEY: str
    user_id: str
    round: Literal[1, 2, 3]
    errors: list[ErrorDispute]

class Letter(BaseModel):
    repo: str
    letter: str

class PersonalInfo(BaseModel):
    first_name: str
    middle_name: str
    last_name: str
    address: str
    city: str
    state: str
    postal_code: str

class GenerateLetterResponse(BaseModel):
    letters: list[Letter]
    sender: PersonalInfo

from utils.get_credit_repo_data import get_credit_repo_data
    
@app.post("/generate-letter")
def generate_letter(request:GenerateLetterRequest) -> GenerateLetterResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    vector_store = get_vector_store(request.user_id)

    results = vector_store.get(
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

    for line in personal_info:
        if "primer nombre" in line:
            first_name = re.search(r": (.+)$", line).group(1)
        elif "segundo nombre" in line:
            match = re.search(r": (.+)$", line)
            if match:
                middle_name = match.group(1)
        elif "apellido" in line:
            last_name = re.search(r": (.+)$", line).group(1)

    full_name = " ".join([first_name, middle_name, last_name]).strip()

    # Obtener fecha actual
    curr_date = date.today().isoformat()  # 'YYYY-MM-DD'

    # Extraer dirección actual
    address = ""
    city = ""
    state = ""
    postal_code = ""
    bdate = ""
    ssn = ""

    for line in personal_info:
        if "Residiendo Actualmente" in line:
            city = re.search(r"Ciudad ([^,;]+)", line).group(1)
            state = re.search(r"Estado (\w{2})", line).group(1)
            postal_code = re.search(r"Codigo Postal (\d+)", line).group(1)
            address = re.search(r"Calle ([^;]+)", line).group(1)
            break
        if "Mi Fecha de nacimiento es: " in line:
            bdate = re.search(r": (.+)$", line).group(1)
        if "Mi SSN es: " in line:
            ssn = re.search(r": (.+)$", line).group(1)

    header = "\n".join([full_name, address, f"{city}, {state} {postal_code}"]) + "\n"

    # DOB: [bdate]    SSN: [ss_number]
    header += f"\nDOB: {bdate}    SSN: {ssn}\n\n"

    footer = f"\nSincerely,\n\n{full_name}"

    llm = ChatOpenAI(model="gpt-5-mini")

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

    letters = []

    for error in error_list:
        if len(error['errors']):
            repo_data = get_credit_repo_data(error['repo'])

            prompt = f"""You are a letter-writing assistant. Given the user's personal information and a list of credit report errors, produce a formal dispute letter:
           
            Write the body of a dispute letter to TransUnion for the {request.round}th round of disputes. 
            Do NOT include any header, footer, contact information, dates, or signatures. 
            Only output the body text of the letter.
            The tone must escalate with each round, so third round must be the most aggressive, the second round must be more aggressive than the first round, and the first round must be the most polite.
            The letter should be written on english.
            The letter should be written in a professional tone.
            The letter is for {error["repo"]} bureau, but dont introduce the letter like Dear bereau or anything like that , its just for you know the context.
            Do not output anything except the completed letter text. Use the following input data:

            Errors: {error['errors']}"""

            response = llm.invoke(prompt)

            letters.append({
                'repo': error['repo'],
                'letter': f'{header}\n{error["repo"]}\n{repo_data["address"]}\n{repo_data["city"]}, {repo_data["state"]}, {repo_data["zip_code"]}\n\nDate:{curr_date}\n\nDear {error["repo"]},\n\n{response.content}\n{footer}'
            })


    return {
        'letters': letters,
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
    Expirian: bool = Field(description="Si esta presente en Expirian")

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

@app.post("/verify-errors")
def verify_errors(request: VerifyErrorsRequest) -> VerifyErrorsResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

    report = get_user_report(request.user_id)

    errors = "\n"
    i = 1

    for error in request.errors:
        errors += f"{i}- "
        errors += get_error_string(error)
        errors += "\n"
        i += 1


    prompt = f"""
        Eres un sistema de verificacion de errores en el credito, debes devolver de manera ordenada si estan aun presentes o no en el credito los siguientes errores, analizalos uno por uno, ten todo en cuenta, y responde por cada error Verdadero(si esta presente el error en el reporte de credito), Falso(si no aparece en el credito), esto por cada buro de credito (Expirian, TransUnion, Equifax), junto con el identificador del Error:
        
        Errores:
        {errors}
        Los informes de los tres burós se encuentran a continuación:
        {report}
    """

    llm = ChatOpenAI(model="gpt-5")
    structured_llm = llm.with_structured_output(VerifyErrorsResponse)
    response = structured_llm.invoke(prompt)
    return response

class CompareErrorsRequest(BaseModel):
    errors_1: list[ErrorDispute]
    errors_2: list[ErrorDisputeWithId]
    API_KEY: str

class CompareErrorsResponse(BaseModel):
    same_errors_ids: list[str]

@app.post("/compare-errors")
def compare_errors(request: CompareErrorsRequest) -> CompareErrorsResponse:
    if os.getenv("API_KEY") != request.API_KEY:
        raise HTTPException(status_code=400, detail="Api key dont match")

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
    response = structured_llm.invoke(prompt)
    return response


# insert_general_knowledge()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
