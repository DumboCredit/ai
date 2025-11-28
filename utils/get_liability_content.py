from utils.get_translation import get_translation
from models import _CREDIT_LIABILITY

def get_late_payments(liability:_CREDIT_LIABILITY) -> str:
    late_payments = 0
    late_payments += float(liability.LATE_COUNT.Days30 or 0) 
    late_payments += float(liability.LATE_COUNT.Days60 or 0) 
    late_payments += float(liability.LATE_COUNT.Days90 or 0) 
    return late_payments

def get_liability_content(libCC:_CREDIT_LIABILITY) -> str:
    """Summary of the liability account"""
    isChargeOff = libCC.IsChargeoffIndicator == "Y"
    isCollection = libCC.IsCollectionIndicator == "Y"
    isCreditCard = libCC.CreditLoanType == "CreditCard" or libCC.CreditLoanType == "ChargeAccount"
    translated_credit_loan_type = get_translation(libCC.CreditLoanType)
    content = f"{translated_credit_loan_type or "Otro prestamo"}: "
    if isChargeOff:
        content += f"Es un charge off (cobranza). "
    if isCollection:
        content += f"Es un collection (cobranza). "
    if libCC.UnpaidBalanceAmount:
        content += f"Saldo: {libCC.UnpaidBalanceAmount}. "
    if not isCreditCard and libCC.UnpaidBalanceAmount:
        base_amount = float(libCC.OriginalBalanceAmount or libCC.HighCreditAmount or libCC.UnpaidBalanceAmount)
        if base_amount != 0:
            content += f"Queda el: {(float(libCC.UnpaidBalanceAmount)/base_amount) *100}% para pagar de este prestamo. "
    if not libCC.CreditLimitAmount is None:
        content += f"Limite crediticio: {libCC.CreditLimitAmount}. "
    if not libCC.LATE_COUNT is None:
        content += f"Pagos atrasados: {get_late_payments(libCC)}. "
        if not libCC.LATE_COUNT.Days30 is None: 
            content += f"Pagos atrasados por 30 dias: {libCC.LATE_COUNT.Days30}. "
        if not libCC.LATE_COUNT.Days60 is None:
            content += f"Pagos atrasados por 60 dias: {libCC.LATE_COUNT.Days60}. "
        if not libCC.LATE_COUNT.Days90 is None:
            content += f"Pagos atrasados por 90 dias: {libCC.LATE_COUNT.Days90}. "
    if not libCC.MonthsReviewedCount is None:
        content += f"Total de pagos: {libCC.MonthsReviewedCount}. "
    if not libCC.MonthlyPaymentAmount is None:
        content += f"Cantidad de Pago Mensual: {libCC.MonthlyPaymentAmount}. "
    if not libCC.CURRENT_RATING is None:
        if not libCC.CURRENT_RATING.Type is None:
            content += f"Estado de la cuenta: {get_translation(libCC.CURRENT_RATING.Type)}. "
        elif not libCC.AccountStatusType is None:
            content += f"Estado de la cuenta: {get_translation(libCC.AccountStatusType)}. "
    if not libCC.IsClosedIndicator is None:
        if libCC.IsClosedIndicator == "Y":
            content += f"La cuenta esta cerrada. "
        else:
            content += f"La cuenta esta abierta. "
    if not libCC.PastDueAmount is None:
        content += f"Importe Vencido: {libCC.PastDueAmount}. "
    if not libCC.AccountIdentifier is None:
        content += f"Numero de cuenta: {libCC.AccountIdentifier}. "
    if not libCC.AccountOpenedDate is None:
        content += f"Fecha de apertura de la cuenta: {libCC.AccountOpenedDate}. "
    if not libCC.LastActivityDate is None:
        content += f"Fecha de ultima actividad: {libCC.LastActivityDate}. "
    if not libCC.AccountOwnershipType is None:
        content += f"Responsabilidad: {get_translation(libCC.AccountOwnershipType)}. "
    if not libCC.MonthsReviewedCount is None:
        content += f"Meses examinados: {libCC.MonthsReviewedCount}. "
    if not libCC.TermsMonthsCount is None:
        content += f"Recuento de Plazos: {libCC.TermsMonthsCount}. "
    if not libCC.TermsSourceType is None:
        content += f"Tipo de fuente de plazo: {get_translation(libCC.TermsSourceType)}. "
    if not libCC.RawIndustryText is None:
        content += f"Tipo de Fuente de Plazo: {libCC.RawIndustryText}. "
    if not libCC.HighCreditAmount is None:
        content += f"Saldo alto: {libCC.HighCreditAmount}. "
    if not libCC.CREDITOR.Name is None:
        content += f"Nombre del acreedor: {libCC.CREDITOR.Name}. "
    if not libCC.CREDITOR.City is None:
        content += f"Ciudad del acreedor: {libCC.CREDITOR.City}. "
    if not libCC.CREDITOR.State is None:
        content += f"Estado del acreedor: {libCC.CREDITOR.State}. "
    if not libCC.CREDITOR.PostalCode is None:
        content += f"Codigo postal del acreedor: {libCC.CREDITOR.PostalCode}. "
    if not libCC.CREDITOR.StreetAddress is None:
        content += f"Direccion del acreedor: {libCC.CREDITOR.StreetAddress}. "
    if not libCC.TradelineHashComplex is None:
        content += f"ID de la cuenta: {libCC.TradelineHashComplex}. "
    if isinstance(libCC.CREDIT_REPOSITORY, list):
        content += f"En los buros de credito: "
        for repo in libCC.CREDIT_REPOSITORY:
            content += f"{repo.SourceType}, "
        content += ". "
    else:
        content += f"Buro de Credito: {libCC.CREDIT_REPOSITORY.SourceType}. "
    return content