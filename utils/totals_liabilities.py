from models import CreditRequest

def get_credit_liability_total(prefix: str, loan_types: list[str], credit_repositories: list[str], historic_credit:CreditRequest) -> str:
    total_content = f"{prefix}: "
    total = 0
    for liability in historic_credit.CREDIT_LIABILITY:
        repo_match = False
        if isinstance(liability.CREDIT_REPOSITORY, list):
            total
        else:
            repo_match = getattr(liability.CREDIT_REPOSITORY, "SourceType", None) in credit_repositories
        loan_type_match = getattr(liability, "CreditLoanType", None) in loan_types
        if repo_match and loan_type_match:
            total+=1

    return f"{total_content} {total}"

def get_credit_cards_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["CreditCard", "ChargeAccount"]
    return get_credit_liability_total("Numero de tarjetas de credito", desired_credit_loan_types, credit_repositories, historic_credit)

def get_auto_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["Automobile", "AutoLoan"]
    return get_credit_liability_total("Numero de prestamos de auto", desired_credit_loan_types, credit_repositories, historic_credit)

def get_education_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["Educational"]
    return get_credit_liability_total("Numero de prestamos estudiantiles", desired_credit_loan_types, credit_repositories, historic_credit)
    
def get_mortgage_loans_content(historic_credit:CreditRequest, credit_repositories: list[str]) -> str:
    desired_credit_loan_types = ["ConventionalRealEstateMortgage"]
    return get_credit_liability_total("Numero de prestamos inmobiliarios", desired_credit_loan_types, credit_repositories, historic_credit)
