from typing import Optional, Union
from pydantic import BaseModel

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

class _LATE_COUNT(BaseModel):
    Days30: Optional[str] = None
    Days60: Optional[str] = None
    Days90: Optional[str] = None

class _HIGHEST_ADVERSE_RATING(BaseModel):
    Type: str

class _CURRENT_RATING(BaseModel):
    Type: str

class _CREDIT_LIABILITY(BaseModel):
    CreditLiabilityID: str
    OriginalBalanceAmount: Optional[str] = None
    UnpaidBalanceAmount: Optional[str] = None
    MonthlyPaymentAmount: Optional[str] = None
    TermsMonthsCount: Optional[str] = None
    MonthsReviewedCount: Optional[str] = None
    CreditLoanType: Optional[str] = None
    CreditLimitAmount: Optional[str] = None
    LATE_COUNT: Optional[_LATE_COUNT] = None
    CREDITOR: _CREDITOR 
    RawIndustryText: Optional[str] = None
    AccountStatusType: Optional[str] = None
    HighCreditAmount: Optional[str] = None
    TermsSourceType: Optional[str] = None
    PAYMENT_PATTERN: _PAYMENT_PATTERN
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