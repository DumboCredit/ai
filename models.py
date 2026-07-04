from typing import Optional, Union, Dict, List
from pydantic import BaseModel

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
    reportedDate: Optional[int] = None
    creditor: Optional[V3Creditor] = None
    type: Optional[str] = None
    bureau: List[str] = []


class V3Collection(BaseModel):
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