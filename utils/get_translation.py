import json

with open("translations.json", "r") as f:
    translations = json.load(f)

def get_translation(text: str) -> str:
    """Get translation for a given text"""
    if text in translations:
        return translations[text]
    else:
        return text