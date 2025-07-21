import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_attributes_from_text(text):
    doc = nlp(text)
    extracted = []

    value_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(EUR|kWh)?", text)

    for sent in doc.sents:
        for match in value_matches:
            number, unit = match
            label = "Unknown"

            if "electricity" in sent.text.lower():
                label = "ElectricityPrice"
            elif "consumption" in sent.text.lower():
                label = "ConsumptionKWh"
            elif "maintenance" in sent.text.lower():
                label = "MaintenanceCost"
            else:
                continue  

            extracted.append({
                "Attribute": label,
                "Value": float(number),
                "Unit": unit
            })
    return extracted
