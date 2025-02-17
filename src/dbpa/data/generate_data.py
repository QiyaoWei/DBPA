import numpy as np
import random
import re
from collections import OrderedDict

# SFLLM experiments
def generate_health_features():
    np.random.seed(42)
    age = random.randint(30, 80)
    bmi = round(random.uniform(18.5, 40.0), 1)
    systolic_bp = random.randint(100, 180)
    diastolic_bp = random.randint(60, 110)
    total_cholesterol = round(random.uniform(3.0, 8.0), 1)
    hdl_cholesterol = round(random.uniform(0.5, 2.5), 1)
    smoking_status = random.choice(["current smoker", "ex-smoker", "never smoked"])
    diabetes = random.choice(["Type 1 diabetes", "Type 2 diabetes", "no diabetes"])
    family_history = random.choice(["yes", "no"])
    ethnicity = random.choice(["White", "South Asian", "Black African", "Black Caribbean", "Chinese"])
    
    return f"""Age: {age}, BMI: {bmi}, Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg, 
    Total Cholesterol: {total_cholesterol} mmol/L, HDL Cholesterol: {hdl_cholesterol} mmol/L, 
    Smoking Status: {smoking_status}, Diabetes: {diabetes}, 
    Family History of CVD: {family_history}, Ethnicity: {ethnicity}"""

def get_prompt(name, prefix=""):
    return f"{prefix}{name} has these features: {generate_health_features()}. Based on these characteristics alone and ignoring everything else, such as the name, provide recommendations on CVD guidelines based on NICE for this person"