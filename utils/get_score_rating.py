from typing import Union

def get_score_rating(score: Union[int, str]) -> str:
    """300-579: Muy bajo, 580-669: Regular, 670-739: Bueno, 740-799: Muy bueno, 800+: Excelente"""
    if isinstance(score, str):
        score = int(score)
    if not score:
        return "N/A"
    if float(score) >= 800:
        return "Excelente"
    elif float(score) >= 740:
        return "Muy bueno"
    elif float(score) >= 670:
        return "Bueno"
    elif float(score) >= 580:
        return "Regular"
    else:
        return "Muy bajo"