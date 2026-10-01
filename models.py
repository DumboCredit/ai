from enum import Enum
from typing import Optional, Union, Dict, List
from pydantic import BaseModel, Field

class _RESIDENCE(BaseModel):
    City: Optional[str] = None
    State: Optional[str] = None
    PostalCode: Optional[str] = None
    StreetAddress: Optional[str] = None
    BorrowerResidencyType: Optional[str] = None

class _BORROWER(BaseModel): 
    FirstName: Optional[str] = None
    MiddleName: Optional[str] = None
    LastName: Optional[str] = None
    SSN: Optional[str] = None
    BirthDate: Optional[str] = None
    RESIDENCE: Union[Optional[list[_RESIDENCE]], Optional[_RESIDENCE]] = None

# credit score
class _FACTOR(BaseModel):
    Code: str
    Text: str
class _POSITIVE_FACTOR(BaseModel):
    Code: str
    Text: str
class _CREDIT_SCORE(BaseModel):
    Date: str
    Value: Optional[Union[int, str]] = None
    CreditRepositorySourceType: str
    RiskBasedPricingMax: Optional[str] = None
    RiskBasedPricingMin: Optional[str] = None
    RiskBasedPricingPercent: Optional[str] = None
    FACTOR: Optional[list[_FACTOR]] = None
    POSITIVE_FACTOR: Optional[list[_POSITIVE_FACTOR]] = None

# credit inquiry
class CREDIT_REPOSITORY(BaseModel):
    SourceType: str
class _CREDIT_INQUIRY(BaseModel):
    PurposeType: Optional[str] = None
    Date: str
    Name: str
    RawIndustryText: Optional[str] = None
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
    City: Optional[str] = None
    State: Optional[str] = None
    PostalCode: Optional[str] = None
    StreetAddress: Optional[str] = None

class _PAYMENT_PATTERN(BaseModel):
    StartDate: str

class _LATE_COUNT(BaseModel):
    Days30: Optional[Union[int, str]] = None
    Days60: Optional[Union[int, str]] = None
    Days90: Optional[Union[int, str]] = None

class _HIGHEST_ADVERSE_RATING(BaseModel):
    Type: str

class _CURRENT_RATING(BaseModel):
    Type: str

class _CREDIT_LIABILITY(BaseModel):
    CreditLiabilityID: str
    OriginalBalanceAmount: Optional[Union[int, str]] = None
    UnpaidBalanceAmount: Optional[Union[int, str]] = None
    MonthlyPaymentAmount: Optional[str] = None
    TermsMonthsCount: Optional[str] = None
    MonthsReviewedCount: Optional[str] = None
    CreditLoanType: Optional[str] = None
    CreditLimitAmount: Optional[str] = None
    LATE_COUNT: Optional[_LATE_COUNT] = None
    CREDITOR: _CREDITOR 
    RawIndustryText: Optional[str] = None
    AccountStatusType: Optional[str] = None
    HighCreditAmount: Optional[Union[int, str]] = None
    TermsSourceType: Optional[str] = None
    PAYMENT_PATTERN: Optional[_PAYMENT_PATTERN] = None
    PastDueAmount: Optional[str] = None
    AccountIdentifier: Optional[str] = None
    TradelineHashComplex: Optional[str] = None
    AccountOpenedDate: Optional[str] = None
    LastActivityDate: Optional[str] = None
    AccountOwnershipType: Optional[str] = None
    CURRENT_RATING: Optional[_CURRENT_RATING] = None
    TermsDescription: Optional[str] = None
    CREDIT_REPOSITORY: Union[CREDIT_REPOSITORY, list[CREDIT_REPOSITORY]] 
    HIGHEST_ADVERSE_RATING: Optional[_HIGHEST_ADVERSE_RATING] = None
    IsChargeoffIndicator: Optional[str] = None
    IsCollectionIndicator: Optional[str] = None
    IsClosedIndicator: Optional[str] = None

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


# %% Lessons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Lesson(BaseModel):
    lesson_id: str
    title: str
    description: str
    level_hint: int   # 1-5, which level this lesson typically targets

class AddLessonRequest(BaseModel):
    API_KEY: str
    lesson: Lesson


# %% Credit Plan %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class WeekTask(BaseModel):
    week: int                       # 1-4
    title: str
    task_type: str                  # "action" | "dispute" | "lesson"
    lesson_id: Optional[str] = None  # only when task_type == "lesson"

class MonthPlan(BaseModel):
    month: int
    title: str
    description: str
    estimated_score_gain: int       # realistic points gained this month
    tasks: list[WeekTask]

class LevelPlan(BaseModel):
    level: int                      # 1-5
    name: str                       # fixed level name
    status: str                     # "completed" | "in_progress" | "locked"
    months: list[MonthPlan]

class CreditPlan(BaseModel):
    levels: list[LevelPlan]

class GeneratePlanRequest(BaseModel):
    API_KEY: str
    user_id: str


# %% Score Simulator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class SimulateActionTypeEnum(str, Enum):
    """Catalogo cerrado de acciones que el motor sabe aplicar al reporte."""
    PAY_DOWN_BALANCE = "pay_down_balance"
    INCREASE_BALANCE = "increase_balance"
    MAX_OUT_CARDS = "max_out_cards"
    CHANGE_CREDIT_LIMIT = "change_credit_limit"
    REMOVE_ACCOUNT = "remove_account"
    REMOVE_INQUIRY = "remove_inquiry"
    REMOVE_LATE_PAYMENTS = "remove_late_payments"
    ADD_LATE_PAYMENT = "add_late_payment"
    OPEN_ACCOUNT = "open_account"
    CLOSE_ACCOUNT = "close_account"
    WAIT_MONTHS = "wait_months"

class SimulatedAction(BaseModel):
    """Una accion estructurada. El LLM solo traduce el texto del usuario a esto;
    los puntos los calcula el motor determinista."""
    type: SimulateActionTypeEnum = Field(description="El tipo de accion, uno de los valores del enum")
    account_ref: Optional[str] = Field(default=None, description="Ref exacto de la cuenta (A1, A2...) tal como aparece en la lista de cuentas del usuario")
    creditor: Optional[str] = Field(default=None, description="Nombre del acreedor si no hay ref, exacto como aparece en la lista")
    inquiry_ref: Optional[str] = Field(default=None, description="Ref exacto de la consulta (I1, I2...) cuando la accion es sobre un inquiry")
    apply_to_all: bool = Field(default=False, description="True cuando la accion aplica a todas las cuentas del tipo indicado, no a una sola")
    account_kind: Optional[str] = Field(default=None, description="revolving | installment | mortgage, para filtrar o para abrir cuenta nueva")
    amount: Optional[float] = Field(default=None, description="Monto en dolares a pagar o a cargar. Vacio = pagar el saldo completo")
    new_limit: Optional[float] = Field(default=None, description="Nuevo limite crediticio, o limite de la cuenta nueva")
    days_late: Optional[int] = Field(default=None, description="30, 60 o 90: severidad del atraso simulado")
    count: Optional[int] = Field(default=None, description="Cuantos items (atrasos, consultas) afecta la accion")
    months: Optional[int] = Field(default=None, description="Meses a avanzar en el tiempo para wait_months")
    note: str = Field(description="En una frase, que se entendio de la peticion del usuario para esta accion")

class SimulatedActionPlan(BaseModel):
    """Lo que el LLM devuelve al interpretar la accion en lenguaje natural."""
    actions: list[SimulatedAction] = Field(description="Acciones en el orden en que deben aplicarse. Vacio si la peticion no se puede representar")
    interpretation: str = Field(description="Resumen en una frase de lo que se va a simular")
    unsupported_reason: Optional[str] = Field(default=None, description="Si no se pudo representar la peticion, por que")

class SimulationAccount(BaseModel):
    """Un tradeline tal como lo ve el simulador, con el ref que aceptan las acciones."""
    ref: str                     # "A7": usarlo como account_ref al simular
    creditor: str
    kind: str                    # revolving | installment | mortgage | other
    is_open: bool
    balance: float
    limit: float
    utilization: Optional[float] = None   # solo revolventes con limite
    is_collection: bool = False
    is_chargeoff: bool = False
    late_payments: int = 0
    opened_at: Optional[str] = None
    bureaus: list[str] = []      # burós que reportan este tradeline

class SimulationInquiry(BaseModel):
    ref: str                     # "I3": usarlo como inquiry_ref al simular
    name: str
    date: Optional[str] = None
    bureaus: list[str] = []

class SimulationOptionsRequest(BaseModel):
    API_KEY: str
    user_id: str

class SimulationOptionsResponse(BaseModel):
    """Todo lo simulable del reporte. Determinista y sin LLM."""
    scores: Dict[str, int]       # {buró: puntaje actual}
    accounts: list[SimulationAccount]
    inquiries: list[SimulationInquiry]

class SimulationNarrative(BaseModel):
    """Lo unico que el LLM escribe al final: texto sobre numeros ya fijos."""
    explanation: str = Field(description="2-3 oraciones en español explicando por que se mueve el puntaje, citando cuentas reales")
    explanation_en: str = Field(description="The same explanation in English, a faithful translation and not a different summary")

class ScoreFactorDelta(BaseModel):
    factor: str              # payment_history | utilization | credit_age | credit_mix | inquiries
    weight: float            # peso FICO del factor
    health_before: float     # 0-1
    health_after: float      # 0-1
    contribution: float      # salud ponderada ganada/perdida por este factor

class BureauScoreImpact(BaseModel):
    bureau: str              # "TransUnion" | "Equifax" | "Experian"
    current_score: int
    estimated_new_score: int
    impact: int              # negativo = pérdida, positivo = ganancia
    impact_min: int          # extremo pesimista de la estimacion
    impact_max: int          # extremo optimista de la estimacion
    factors: list[ScoreFactorDelta] = []
    notes: list[str] = []    # p.ej. que este buró no reporta la cuenta simulada

class SimulateScoreRequest(BaseModel):
    API_KEY: str
    user_id: str
    action: str = ""  # lenguaje natural: "dejar que mi cuenta de Capital One se venza 2 meses"
    # El front puede mandar las acciones ya estructuradas (p.ej. togglear un item
    # negativo del reporte) y entonces no se llama al LLM interpretador.
    actions: Optional[list[SimulatedAction]] = None
    # False salta el LLM narrador y deja la respuesta 100% determinista: para
    # sliders y toggles que se recalculan en cada interaccion.
    explain: bool = True

class SimulateScoreResponse(BaseModel):
    action: str
    interpreted_as: str      # que entendio el sistema que se estaba simulando
    actions: list[SimulatedAction] = []
    impacts: list[BureauScoreImpact]
    explanation: str         # 2-3 oraciones en español explicando el impacto
    explanation_en: str      # la misma explicación en inglés
    risk_level: str          # "low" | "medium" | "high" | "critical"
    caveats: list[str] = []      # supuestos y limitaciones, en español
    caveats_en: list[str] = []   # los mismos, en inglés
    estimate_basis: str = (
        "Estimación propia de Dumbo basada en los factores FICO y en tu reporte real. "
        "No es tu puntaje FICO ni VantageScore oficial: es orientación, no un pronóstico exacto."
    )
    estimate_basis_en: str = (
        "Dumbo's own estimate based on FICO factors and your actual report. "
        "It is not your official FICO or VantageScore: it is guidance, not an exact forecast."
    )


# %% Credit Report v3 (estructura nueva, buró Equifax 3B) %%%%%%%%%%%%%%%%%%%%%%
# Modelos que espejan la interfaz `CreditReport` de dumbo-prod (src/types/userTypes
# + src/utils/equifaxCreditReport.ts). Los Record<CREDIT_REPO, ...> del TS llegan
# como diccionarios cuyas claves son "TransUnion" | "Experian" | "Equifax".
# Todo es Optional para tolerar reportes parciales sin romper la ingesta.

# Claves de buró tal como las serializa el enum CREDIT_REPO del front.
BUREAU_KEYS = ["Equifax", "Experian", "TransUnion"]


class V3Address(BaseModel):
    country: Optional[str] = None
    postalCode: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None


class V3Creditor(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[V3Address] = None


class V3CreditMonth(BaseModel):
    monthType: Optional[str] = None
    value: Optional[str] = None
    label: Optional[str] = None


class V3PaymentHistoryYear(BaseModel):
    year: Optional[int] = None
    months: List[V3CreditMonth] = []


class V3Account(BaseModel):
    # Identificador estable: el mismo valor en los tres burós para la misma cuenta.
    # Opcional porque dumbo-prod todavía puede no mandarlo; sin él, el simulador
    # deduce la agrupación por apertura + saldo + límite.
    id: Optional[str] = None
    number: Optional[str] = None
    name: Optional[str] = None
    isOpen: Optional[bool] = None
    amount: Optional[float] = None
    creditLimit: Optional[float] = None
    highCredit: Optional[float] = None
    openedAt: Optional[int] = None
    closedAt: Optional[int] = None
    status: Optional[str] = None
    percentage: Optional[float] = None
    monthlyPayment: Optional[float] = None
    loanType: Optional[str] = None
    lastActivityAt: Optional[int] = None
    responsability: Optional[str] = None
    creditor: Optional[V3Creditor] = None
    monthsReviewed: Optional[int] = None
    percentagePaymentsOnTime: Optional[float] = None
    pastDueAmount: Optional[float] = None
    paymentHistory: List[V3PaymentHistoryYear] = []
    paymentStatus: Optional[str] = None
    bureau: List[str] = []


class V3AccountsByType(BaseModel):
    creditCards: List[V3Account] = []
    educationalLoans: List[V3Account] = []
    mortagageLoans: List[V3Account] = []  # (typo intencional: coincide con el TS)
    autoLoans: List[V3Account] = []


class V3Inquiry(BaseModel):
    id: Optional[str] = None  # igual entre burós para la misma consulta; opcional
    reportedDate: Optional[int] = None
    creditor: Optional[V3Creditor] = None
    type: Optional[str] = None
    bureau: List[str] = []


class V3Collection(BaseModel):
    id: Optional[str] = None  # igual entre burós para la misma cobranza; opcional
    accountNumber: Optional[str] = None
    agencyClient: Optional[V3Creditor] = None
    originalCreditor: Optional[V3Creditor] = None
    status: Optional[str] = None
    amount: Optional[float] = None
    reportedDate: Optional[int] = None
    bureau: List[str] = []


class V3PublicRecord(BaseModel):
    refNumber: Optional[str] = None
    status: Optional[str] = None
    courtName: Optional[str] = None
    reportedDate: Optional[int] = None
    filedDate: Optional[int] = None
    assetAmount: Optional[float] = None
    amount: Optional[float] = None
    type: Optional[str] = None
    bureau: List[str] = []


class V3Summary(BaseModel):
    totalAccounts: Optional[int] = None
    totalOpenAccounts: Optional[int] = None
    totalClosedAccounts: Optional[int] = None
    totalCollections: Optional[int] = None
    totalPublicRecords: Optional[int] = None
    totalInquiries: Optional[int] = None
    totalCreditCards: Optional[int] = None
    totalMortage: Optional[int] = None
    totalAuto: Optional[int] = None
    totalEducational: Optional[int] = None
    totalOtherAccounts: Optional[int] = None


class V3PersonalInfo(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    middleName: Optional[str] = None
    currentAddress: Optional[V3Address] = None
    homePhone: Optional[str] = None
    mobilePhone: Optional[str] = None
    nationalIdentifier: Optional[str] = None
    dateOfBirth: Optional[Union[int, str]] = None


class V3ScoreHistoryPoint(BaseModel):
    reportedDate: Optional[int] = None
    value: Optional[float] = None


class CreditReportV3(BaseModel):
    accounts: Dict[str, V3AccountsByType] = {}
    inquiries: Dict[str, List[V3Inquiry]] = {}
    summary: Dict[str, V3Summary] = {}
    personalInfo: Dict[str, V3PersonalInfo] = {}
    creditors: Dict[str, List[V3Creditor]] = {}
    collections: Dict[str, List[V3Collection]] = {}
    publicRecords: Dict[str, List[V3PublicRecord]] = {}
    scores: Dict[str, Optional[float]] = {}
    scoreHistory: Dict[str, List[V3ScoreHistoryPoint]] = {}
    generatedDate: Optional[int] = None


class AddUserCreditDataV3Request(BaseModel):
    """Request del endpoint /add-user-credit-data-v3.

    `data` es el CreditReport (estructura v3) serializado como JSON y cifrado con
    AES-256-GCM (formato base64 "IV:EncryptedData:AuthTag"), el mismo esquema que
    usa dumbo-prod. USER_ID y API_KEY viajan en claro para autenticar y enrutar.
    """
    API_KEY: str
    USER_ID: str
    data: str