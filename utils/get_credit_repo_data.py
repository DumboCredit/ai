def get_credit_repo_data(credit_repo: str) -> str:
    if credit_repo == "Equifax":
        return {
            'address': 'P.O. Box 740251',
            'city': 'Atlanta',
            'state': 'GA',
            'zip_code': '30374'
        }
    elif credit_repo == "Experian":
        return {
            'address': 'P.O. Box 9534',
            'city': 'Allen',
            'state': 'TX',
            'zip_code': '75013'
        }
    elif credit_repo == "TransUnion":
        return {
            'address': 'P.O. Box 2000',
            'city': 'Chester',
            'state': 'PA',
            'zip_code': '19016'
        }
    else:
        return None