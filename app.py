import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from scipy import stats
from statsmodels.stats.proportion import proportion_confint 
from clustering import clustering_page
from predictive import predictive_page

st.set_page_config(page_title="Patient Data Analysis", layout="wide")

class DataProcessor:
    @staticmethod
    def calculate_age(row):
        if pd.isna(row["RIDAGEYR"]) and pd.isna(row["RIDAGEMN"]):
            return None
        # For infants, show age in fraction of years
        if row["RIDAGEYR"] == 0:
            return round(row["RIDAGEMN"] / 12, 2)
        else:
            return float(row["RIDAGEYR"])

    @staticmethod
    @st.cache_data
    def load_demographics(filepath="data/demographic.csv"):
        df = pd.read_csv(filepath, usecols=["SEQN", "RIAGENDR", "RIDAGEYR", "RIDAGEMN", "WTMEC2YR", "DMDEDUC2", "INDFMPIR"])
        df["RIAGENDR"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
        df["Age"] = df.apply(DataProcessor.calculate_age, axis=1)
        if "DMDEDUC2" in df.columns:
                edu_map = {
                    1: "Less than 9th grade",
                    2: "9-11th grade (No diploma)", 
                    3: "High school graduate/GED",
                    4: "Some college or AA degree",
                    5: "College graduate or above",
                    7: np.nan, # Refused
                    9: np.nan  # Don't know
                }

                df["Education Level (Adults 20+)"] = df["DMDEDUC2"].map(edu_map)
            
        if "INDFMPIR" in df.columns:
                df["PIR_Ratio"] = pd.to_numeric(df["INDFMPIR"], errors='coerce')
                pir_bins = [-np.inf, 0.999, 1.999, 3.999, np.inf] 
                pir_labels = ["PIR <1.0 (Below Poverty)", 
                              "PIR 1.0-1.99 (Low Income)", 
                              "PIR 2.0-3.99 (Middle Income)", 
                              "PIR >=4.0 (Higher Income)"]
                df["Income_to_Poverty_Category"] = pd.cut(df["PIR_Ratio"], bins=pir_bins, labels=pir_labels, right=True, include_lowest=True)

        cols_to_drop_original = ["RIDAGEMN", "RIDAGEYR", "DMDEDUC2", "INDFMPIR"]
        df.drop(columns=[col for col in cols_to_drop_original if col in df.columns], inplace=True)
            
           
        return df


    @staticmethod
    @st.cache_data
    def load_diet(filepath="data/diet.csv"):
        df = pd.read_csv(filepath)
        selected_columns = [
            "SEQN", "DBD100", "DRQSPREP", "DRQSDIET", "DRQSDT1", "DRQSDT2",
            "DRQSDT3", "DRQSDT4", "DRQSDT5", "DRQSDT6", "DRQSDT7", "DRQSDT8", "DRQSDT9",
            "DRQSDT10", "DRQSDT12", "DRQSDT91", "DR1TNUMF", "DR1TKCAL", "DR1TPROT",
            "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT", "DR1TMFAT",
            "DR1TPFAT", "DR1TCHOL", "DR1TATOC", "DR1TATOA", "DR1TRET", "DR1TVARA",
            "DR1TACAR", "DR1TBCAR", "DR1TCRYP", "DR1TLYCO", "DR1TLZ", "DR1TVB1",
            "DR1TVB2", "DR1TNIAC", "DR1TVB6", "DR1TFOLA", "DR1TFA", "DR1TFF",
            "DR1TFDFE", "DR1TCHL", "DR1TVB12", "DR1TB12A", "DR1TVC", "DR1TVD",
            "DR1TVK", "DR1TCALC", "DR1TPHOS", "DR1TMAGN", "DR1TIRON", "DR1TZINC",
            "DR1TCOPP", "DR1TSODI", "DR1TPOTA", "DR1TSELE", "DR1TCAFF", "DR1TTHEO",
            "DR1TALCO"
        ]
        df = df[selected_columns]
        diet_replace_dict = {
            "DBD100": {1: "Rarely", 2: "Occasionally", 3: "Very often", 7: "Refused", 9: "Don't know"},
            "DRQSPREP": {1: "Never", 2: "Rarely", 3: "Occasionally", 4: "Very often", 9: "Don't know"},
            "DRQSDIET": {1: "Yes", 2: "No", 9: "Don't know"},
            "DRQSDT1": {1: "Yes"},
            "DRQSDT2": {2: "Yes"},
            "DRQSDT3": {3: "Yes"},
            "DRQSDT4": {4: "Yes"},
            "DRQSDT5": {5: "Yes"},
            "DRQSDT6": {6: "Yes"},
            "DRQSDT7": {7: "Yes"},
            "DRQSDT8": {8: "Yes"},
            "DRQSDT9": {9: "Yes"},
            "DRQSDT10": {10: "Yes"},
            "DRQSDT12": {12: "Yes"},
            "DRQSDT91": {91: "Yes"}
        }
        df = df.replace(diet_replace_dict)
        rename_dict = {
            "DBD100": "How often do you put salt on the food at the table?",
            "DRQSPREP": "Salt used in preparation",
            "DRQSDIET": "On special diet",
            "DRQSDT1": "Weight loss or low calorie diet",
            "DRQSDT2": "Low fat/Low cholesterol diet",
            "DRQSDT3": "Low salt/Low sodium diet",
            "DRQSDT4": "Sugar free/Low sugar diet",
            "DRQSDT5": "Low fiber diet",
            "DRQSDT6": "High fiber diet",
            "DRQSDT7": "Diabetic diet",
            "DRQSDT8": "Weight gain/Muscle building diet",
            "DRQSDT9": "Low carbohydrate diet",
            "DRQSDT10": "High protein diet",
            "DRQSDT12": "Renal/Kidney diet",
            "DRQSDT91": "Other special diet",
            "DR1TNUMF": "Number of foods reported",
            "DR1TKCAL": "Energy (kcal)",
            "DR1TPROT": "Protein (gm)",
            "DR1TCARB": "Carbohydrate (gm)",
            "DR1TSUGR": "Total sugars (gm)",
            "DR1TFIBE": "Dietary fiber (gm)",
            "DR1TTFAT": "Total fat (gm)",
            "DR1TSFAT": "Total saturated fatty acids (gm)",
            "DR1TMFAT": "Total monounsaturated fatty acids (gm)",
            "DR1TPFAT": "Total polyunsaturated fatty acids (gm)",
            "DR1TCHOL": "Cholesterol (mg)",
            "DR1TATOC": "Vitamin E as alpha-tocopherol (mg)",
            "DR1TATOA": "Added alpha-tocopherol (Vitamin E) (mg)",
            "DR1TRET": "Retinol (mcg)",
            "DR1TVARA": "Vitamin A, RAE (mcg)",
            "DR1TACAR": "Alpha-carotene (mcg)",
            "DR1TBCAR": "Beta-carotene (mcg)",
            "DR1TCRYP": "Beta-cryptoxanthin (mcg)",
            "DR1TLYCO": "Lycopene (mcg)",
            "DR1TLZ": "Lutein + zeaxanthin (mcg)",
            "DR1TVB1": "Thiamin (Vitamin B1) (mg)",
            "DR1TVB2": "Riboflavin (Vitamin B2) (mg)",
            "DR1TNIAC": "Niacin (mg)",
            "DR1TVB6": "Vitamin B6 (mg)",
            "DR1TFOLA": "Total folate (mcg)",
            "DR1TFA": "Folic acid (mcg)",
            "DR1TFF": "Food folate (mcg)",
            "DR1TFDFE": "Folate, DFE (mcg)",
            "DR1TCHL": "Total choline (mg)",
            "DR1TVB12": "Vitamin B12 (mcg)",
            "DR1TB12A": "Added vitamin B12 (mcg)",
            "DR1TVC": "Vitamin C (mg)",
            "DR1TVD": "Vitamin D (D2 + D3) (mcg)",
            "DR1TVK": "Vitamin K (mcg)",
            "DR1TCALC": "Calcium (mg)",
            "DR1TPHOS": "Phosphorus (mg)",
            "DR1TMAGN": "Magnesium (mg)",
            "DR1TIRON": "Iron (mg)",
            "DR1TZINC": "Zinc (mg)",
            "DR1TCOPP": "Copper (mg)",
            "DR1TSODI": "Sodium (mg)",
            "DR1TPOTA": "Potassium (mg)",
            "DR1TSELE": "Selenium (mcg)",
            "DR1TCAFF": "Caffeine (mg)",
            "DR1TTHEO": "Theobromine (mg)",
            "DR1TALCO": "Alcohol (gm)"
        }
        df = df.rename(columns=rename_dict)
        return df

    @staticmethod
    @st.cache_data
    def load_examination(filepath="data/examination.csv"):
        df = pd.read_csv(filepath)
        selected_columns = [
            "SEQN", "PEASCST1", "PEASCTM1", "BPXPULS", "BPXSY1", "BPXDI1", 
            "BMXWT", "BMXHT", "BMXBMI", "BMDBMIC", "BMXWAIST"
        ]
        df = df[selected_columns]
        examination_replace_dict = {
            "PEASCST1": {1: "Complete", 2: "Partial", 3: "Not done"},
            "BPXPULS": {1: "Regular", 2: "Irregular"},
            "BMDBMIC": {1: "Underweight", 2: "Normal weight", 3: "Overweight", 4: "Obese"}
        }
        df = df.replace(examination_replace_dict)
        rename_dict = {
            "PEASCST1": "Blood Pressure Status",
            "PEASCTM1": "Blood Pressure Time in Seconds",
            "BPXPULS": "Pulse Regular or Irregular",
            "BPXSY1": "Systolic Blood Pressure (1st reading) mm Hg",
            "BPXDI1": "Diastolic Blood Pressure (1st reading) mm Hg",
            "BMXWT": "Weight (kg)",
            "BMXHT": "Standing Height (cm)",
            "BMXBMI": "Body Mass Index (kg/m²)",
            "BMDBMIC": "BMI Category (Children/Adolescents)",
            "BMXWAIST": "Waist Circumference (cm)"
        }
        df = df.rename(columns=rename_dict)
        return df

    @staticmethod
    @st.cache_data
    def load_labs(filepath="data/labs.csv"):
        df = pd.read_csv(filepath)
        selected_columns = [
            "SEQN", "URXUMA", "URXUCR", "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN",
            "LBDHDD", "LBXTR", "LBDLDL", "LBXTC", "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", 
            "LBXEOPCT", "LBXBAPCT", "LBDLYMNO", "LBDMONO", "LBDNENO", "LBDEONO", "LBDBANO", 
            "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMC", "LBXMCHSI", "LBXRDW", 
            "LBXPLTSI", "LBXMPSI", "LBXIN",    
                "LBDINSI", 
                "LBXGLT",   
                "LBDGLTSI" , "LBXGH"
        ]
        df = df[selected_columns]
        labs_replace_dict = {
            "LBXWBCSI": {0: "Missing", 1134: "Missing", 400: "Above Range", "": "Empty Data"},
            "LBXLYPCT": {1145: "Missing"},
            "LBXMOPCT": {1145: "Missing"},
            "LBXNEPCT": {1145: "Missing"},
            "LBXEOPCT": {1145: "Missing"},
            "LBXBAPCT": {1145: "Missing"},
            "LBDLYMNO": {1145: "Missing"},
            "LBDMONO": {1145: "Missing"},
            "LBDNENO": {1145: "Missing"},
            "LBDEONO": {1145: "Missing"},
            "LBDBANO": {1145: "Missing"},
            "LBXHGB": {1134: "Missing"},
            "LBXHCT": {1134: "Missing"},
            "LBXMCVSI": {1134: "Missing"},
            "LBXMC": {1134: "Missing"},
            "LBXMCHSI": {1134: "Missing"},
            "LBXRDW": {1134: "Missing"},
            "LBXPLTSI": {1134: "Missing"},
            "LBXMPSI": {1134: "Missing"},
        }
        df = df.replace(labs_replace_dict)
        rename_dict = {
            "URXUMA": "Albumin, urine (ug/mL)",
            "URXUCR": "Creatinine, urine (mg/dL)",
            "LBXBPB": "Blood lead (ug/dL)",
            "LBXBCD": "Blood cadmium (ug/L)",
            "LBXTHG": "Blood mercury, total (ug/L)",
            "LBXBSE": "Blood selenium (ug/L)",
            "LBXBMN": "Blood manganese (ug/L)",
            "LBDHDD": "Direct HDL-Cholesterol (mg/dL)",
            "LBXTR": "Triglyceride (mg/dL)",
            "LBDLDL": "LDL-cholesterol (mg/dL)",
            "LBXTC": "Total cholesterol (mg/dL)",
            "LBXWBCSI": "White blood cell count (1000 cells/uL)",
            "LBXLYPCT": "Lymphocyte percent (%)",
            "LBXMOPCT": "Monocyte percent (%)",
            "LBXNEPCT": "Segmented neutrophils percent (%)",
            "LBXEOPCT": "Eosinophils percent (%)",
            "LBXBAPCT": "Basophils percent (%)",
            "LBDLYMNO": "Lymphocyte number (1000 cells/uL)",
            "LBDMONO": "Monocyte number (1000 cells/uL)",
            "LBDNENO": "Segmented neutrophils number (1000 cells/uL)",
            "LBDEONO": "Eosinophils number (1000 cells/uL)",
            "LBDBANO": "Basophils number (1000 cells/uL)",
            "LBXRBCSI": "Red blood cell count (million cells/uL)",
            "LBXHGB": "Hemoglobin (g/dL)",
            "LBXHCT": "Hematocrit (%)",
            "LBXMCVSI": "Mean cell volume (fL)",
            "LBXMC": "Mean Cell Hgb Conc. (g/dL)",
            "LBXMCHSI": "Mean cell hemoglobin (pg)",
            "LBXRDW": "Red cell distribution width (%)",
            "LBXPLTSI": "Platelet count (1000 cells/uL)",
            "LBXMPSI": "Mean platelet volume (fL)",
            "LBXIN": "Insulin (uU/mL)",
            "LBDINSI": "Insulin (pmol/L)",
            "LBXGLT": "Two Hour Glucose (OGTT) (mg/dL)",
            "LBDGLTSI": "Two Hour Glucose (OGTT) (mmol/L)",
            "LBXGH": "Glycohemoglobin (HbA1c) (%)"
        }
        df = df.rename(columns=rename_dict)
        return df

    @staticmethod
    @st.cache_data
    def load_medication(filepath="data/medications.csv"):
        try:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
            selected_columns = [
                "SEQN", "RXDUSE", "RXDDRUG", "RXDDAYS", "RXDRSC1", "RXDRSC2", "RXDRSC3", 
                "RXDRSD1", "RXDRSD2", "RXDRSD3", "RXDCOUNT"
            ]
            df = df[selected_columns]
            medication_replace_label = {"RXDUSE": {1: "Yes", 2: "No"}}
            df = df.replace(medication_replace_label)
            rename_dict = {
                "RXDUSE": "Used Medication (30d)",
                "RXDDRUG": "Medication Name",
                "RXDDAYS": "Duration of Use",
                "RXDRSC1": "ICD-10 Code 1",
                "RXDRSC2": "ICD-10 Code 2",
                "RXDRSC3": "ICD-10 Code 3",
                "RXDRSD1": "ICD-10 Description 1",
                "RXDRSD2": "ICD-10 Description 2",
                "RXDRSD3": "ICD-10 Description 3",
                "RXDCOUNT": "Number of Medicines"
            }
            df = df.rename(columns=rename_dict)

            if 'SEQN' in df.columns and "Number of Medicines" in df.columns:
                cols_to_keep_meds = ['SEQN', 'Number of Medicines']
                if 'Used Medication (30d)' in df.columns:
                    cols_to_keep_meds.append('Used Medication (30d)')

                df_unique_meds = df[cols_to_keep_meds].copy()
                df_unique_meds = df_unique_meds.drop_duplicates(subset=['SEQN'], keep='first')
                df = df_unique_meds
            elif 'SEQN' in df.columns:
                df = df.drop_duplicates(subset=['SEQN'], keep='first')

            return df

        except FileNotFoundError:
            print(f"ERROR: Medication file not found: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: Error loading medication data: {e}")
            return pd.DataFrame()


    @staticmethod
    @st.cache_data
    def load_questionnaire(filepath="data/questionnaire.csv"):
        df = pd.read_csv(filepath)
        selected_columns = [
            "SEQN", "DBQ700", "DBQ197", "DBQ223A", "DBQ223B", "DBQ223C", "DBQ223D", "DBQ223E", 
            "DBQ223U", "DBQ229", "DBQ235A", "DBQ235B", "DBQ235C", "DBD895", "DBD900", "DBD905", 
            "DBD910", "FSD032C", "HUQ010", "HUQ020", "HEQ010", "DIQ010", "DIQ160", "DIQ170",
            "BPQ020", "BPQ080", "MCQ010", "MCQ080", "MCQ082", "MCQ220", "MCQ053", "MCQ070","MCQ160A", "MCQ160N", "MCQ160B", "MCQ160C",
            "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160M", 
            "MCQ160L", "MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D", "PAQ605", "PAQ620", "PAQ650", 
            "PAD615", "PAD630", "PAD645", "PAD660", "PAD675",
            "PAD680", "SLD010H", "SMQ040"
        ]
        df = df[selected_columns]
        questionnaire_replace_dict = {
            "DBQ700": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBQ197": {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Varied", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBQ223A": {10: "Yes", 77: "Refused", 99: "Don't know", "": "Missing"},
            "DBQ223B": {11: "Yes", "": "Missing"},
            "DBQ223C": {12: "Yes", "": "Missing"},
            "DBQ223D": {13: "Yes", "": "Missing"},
            "DBQ223E": {14: "Yes", "": "Missing"},
            "DBQ223U": {30: "Yes", "": "Missing"},
            "DBQ229": {1: "Yes", 2: "No", 3: "Varied", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBQ235A": {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Varied", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBQ235B": {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Varied", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBQ235C": {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Varied", 7: "Refused", 9: "Don't know", "": "Missing"},
            "DBD895": {0: "None", 5555: "More than 21", 7777: "Refused", 9999: "Don't know", "": "Missing"},
            "DBD900": {0: "None", 5555: "More than 21", 7777: "Refused", 9999: "Don't know", "": "Missing"},
            "DBD905": {0: "Never", 6666: "More than 90", 7777: "Refused", 9999: "Don't know", "": "Missing"},
            "DBD910": {0: "Never", 6666: "More than 90", 7777: "Refused", 9999: "Don't know", "": "Missing"},
            "FSD032C": {1: "Often", 2: "Sometimes", 3: "Never", 7: "Refused", 9: "Don't know", "": "Missing"},
            "HUQ010": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor", 7: "Refused", 9: "Don't know", "": "Missing"},
            "HUQ020": {1: "Better", 2: "Worse", 3: "Same", 7: "Refused", 9: "Don't know", "": "Missing"},
            "HEQ010": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "DIQ010": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "DIQ160": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "DIQ170": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "BPQ020": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "BPQ080": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ010": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ080": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ082": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ220": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ053": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ070": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ160A":{1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ160N" :{1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ160B" : {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ160C" : {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"},
            "MCQ160D": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", ".": "Missing"},
            "MCQ160E": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", ".": "Missing"},
            "MCQ160F": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", ".": "Missing"},
            "MCQ160M": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", ".": "Missing"},   
            "MCQ160L": {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know", ".": "Missing"},
            "MCQ230A": {},
            "MCQ230B": {},
            "MCQ230C": {},
            "MCQ230D": {},
            "PAQ605": {1: "Yes", 2: "No", 7: np.nan, 9: np.nan, ".": np.nan},
            "PAQ620": {1: "Yes", 2: "No", 7: np.nan, 9: np.nan, ".": np.nan},
            "PAQ650": {1: "Yes", 2: "No", 7: np.nan, 9: np.nan, ".": np.nan},
            "PAD615": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "PAD630": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "PAD645": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "PAD660": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "PAD675": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "PAD680": {7777: np.nan, 9999: np.nan, ".": np.nan}, 
            "SLD010H": {77: np.nan, 99: np.nan, ".": np.nan},   
            "SMQ040": {1: "Every day smoker", 2: "Some days smoker", 3: "Not at all smoker", 7: np.nan, 9: np.nan, ".": np.nan}
                           
        }
        df = df.replace(questionnaire_replace_dict)
        rename_dict = {
            "DBQ700": "Diet Health",
            "DBQ197": "Milk Consumption (30d)",
            "DBQ223A": "Whole Milk",
            "DBQ223B": "2% Milk",
            "DBQ223C": "1% Milk",
            "DBQ223D": "Skim Milk",
            "DBQ223E": "Soy Milk",
            "DBQ223U": "Other Milk",
            "DBQ229": "Regular Milk Drink",
            "DBQ235A": "Milk (Age 5-12)",
            "DBQ235B": "Milk (Age 13-17)",
            "DBQ235C": "Milk (Age 18-35)",
            "DBD895": "Non-home Meals",
            "DBD900": "Fast Food Meals",
            "DBD905": "Ready-to-eat Meals",
            "DBD910": "Frozen Meals",
            "FSD032C": "Affordable Meals",
            "HUQ010": "Health Condition",
            "HUQ020": "Health Change (1yr)",
            "HEQ010": "Ever have HepB",
            "DIQ010": "Ever have Diabetes",
            "DIQ160": "Ever have Prediabetes",
            "DIQ170": "Ever have DiabetesRisk",
            "BPQ020": "Ever have Hypertension",
            "BPQ080": "Ever have HighChol",
            "MCQ010": "Ever have Asthma",
            "MCQ080": "Ever have Overweight",
            "MCQ082": "Ever have Celiac",
            "MCQ220": "Ever have Cancer",
            "MCQ053": "Ever have Anemia",
            "MCQ070": "Ever have Psoriasis",
            "MCQ160A": "Ever have Arthritis",
            "MCQ160N": "Ever have Gout",
            "MCQ160B": "Ever have Congestive Heart Failure",
            "MCQ160C": "Ever have Coronary Heart Disease",
            "MCQ160D": "Ever have Angina",
            "MCQ160E": "Ever have Heart Attack",
            "MCQ160F": "Ever have Stroke",
            "MCQ160M": "Ever have Thyroid Problem",
            "MCQ160L": "Ever have Liver Condition",
            "PAQ605": "Vigorous Work Activity (Yes/No)", 
            "PAQ620": "Moderate Work Activity (Yes/No)",
            "PAQ650": "Vigorous Recreational Activity (Yes/No)",            
            "PAD615": "Mins Vigorous Work Activity",
            "PAD630": "Mins Moderate Work Activity",
            "PAD645": "Mins Walk/Bike Transport",
            "PAD660": "Mins Vigorous Rec Activity",
            "PAD675": "Mins Moderate Rec Activity",           
            "PAD680": "Minutes Sedentary Activity per Day",
            "SLD010H": "Hours Sleep Weekdays",
            "SMQ040": "Smoking Status Now"

            
        }
        df = df.rename(columns=rename_dict)
        return df

    @staticmethod
    @st.cache_data
    def merge_data():
        demo = DataProcessor.load_demographics()
        diet = DataProcessor.load_diet()
        exam = DataProcessor.load_examination()
        labs = DataProcessor.load_labs()
        med = DataProcessor.load_medication()
        quest = DataProcessor.load_questionnaire()
        merged = (demo.merge(diet, on="SEQN", how="outer")
                    .merge(exam, on="SEQN", how="outer")
                    .merge(labs, on="SEQN", how="outer")
                    .merge(quest, on="SEQN", how="outer")
                    .merge(med, on="SEQN", how="outer"))
        return merged


    def load_detailed_medication_records(filepath="data/medications.csv"):
            try:
                df = pd.read_csv(filepath, encoding='ISO-8859-1')

                cols_to_select = ['SEQN', 'RXDDRUG'] 
                df_detailed = df[[col for col in cols_to_select if col in df.columns]].copy()
                df_detailed = df_detailed.rename(columns={"RXDDRUG": "Medication Name"})      
                df_detailed.dropna(subset=["Medication Name"], inplace=True) 
                return df_detailed
            except FileNotFoundError:
                print(f"ERROR: Detailed medication file not found: {filepath}")
                return pd.DataFrame()
            except Exception as e:
                print(f"ERROR: Error loading detailed medication data: {e}")
                return pd.DataFrame()

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Patient Data","Graphs & Trends", "Cluster Profiles",  "Predictive Modeling"])

# Load the preprocessed data (cached so it doesn't run on every page reload)
demographics = DataProcessor.load_demographics()
diet_df = DataProcessor.load_diet()
examination_df = DataProcessor.load_examination()
labs_df = DataProcessor.load_labs()
medication_df = DataProcessor.load_detailed_medication_records()
questionnaire_df = DataProcessor.load_questionnaire()
merged_df = DataProcessor.merge_data()



#  HOME PAGE
if section == "Home":
    st.title("Analyzing Patient Data to Identify Trends in Health Outcomes")
    st.markdown("""
    This application provides an interactive exploration of the research conducted for the bachelor thesis, *"Analyzing Patient Data to Identify Trends in Health Outcomes."*
    It uses the National Health and Nutrition Examination Survey (NHANES) dataset to investigate how lifestyle factors, dietary patterns, and clinical measurements influence health.
    """)

    st.markdown("---")
    st.header("Core Analytical Components")
    st.markdown("Navigate using the sidebar to delve into:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" Exploratory Data Analysis")
        st.markdown("""
        Visualize key trends, demographic distributions, reported dietary habits, and the prevalence of various health conditions within the NHANES sample. This section lays the groundwork by highlighting initial patterns and associations.
        """)

    with col2:
        st.subheader(" Lifestyle Profile Identification")
        st.markdown("""
        Discover how unsupervised clustering techniques were applied to a comprehensive set of behavioral and dietary features. The goal was to identify distinct, data-driven lifestyle profiles among the surveyed adults and understand their characteristics.
        """)


    with col3:
        st.subheader(" Predictive Modeling for Health Risks")
        st.markdown("""
        Investigate the machine learning models developed to predict the risk of significant chronic conditions – specifically diabetes and hypertension. Explore how SHAP values help interpret these models and identify key influencing factors.
        """)


    st.markdown("---")

    st.subheader("About the NHANES Dataset")
    st.markdown("""
    The analyses presented here utilize data from the [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/about/), a vital program by the U.S. Centers for Disease Control and Prevention (CDC). NHANES provides a comprehensive snapshot of the health and nutritional status of the U.S. population through:
    * **Demographics:** Age, gender, socioeconomic data.
    * **Dietary Intake:** Detailed information on food and nutrient consumption.
    * **Physical Examinations:** Clinical measurements like BMI and blood pressure.
    * **Laboratory Tests:** Results for various biochemical markers.
    * **Questionnaires:** Self-reported health status, lifestyle behaviors, and medical history.
    * **Medication Usage:** Data on prescribed medications.
    """)

    st.markdown("""
    The insights derived from these analyses aim to contribute to a more nuanced understanding of population health, demonstrating the power of data-driven approaches to inform evidence-based public health strategies and potentially guide more personalized healthcare interventions.
    """)


# PATIENT DATA PAGE
elif section == "Patient Data":
    st.title("Patient Data Explorer")
    display_df = merged_df.copy() if not merged_df.empty else pd.DataFrame()

    sub_category = st.selectbox("Select Data Category to View:", 
                                ["Merged Data Overview", "Demographics", "Dietary Intake", 
                                 "Examination", "Laboratory Results", "Questionnaire Data", "Medication Data"])
    
    if not display_df.empty:
        st.write(f"Total records in initially merged dataset: {len(display_df)}")
        
        if sub_category == "Merged Data Overview":
            st.header("Merged Data Overview")
            st.dataframe(merged_df)
            st.write(f"Shape of merged data: {display_df.shape}")
        elif sub_category == "Demographics" and not demographics.empty:
            st.header("Demographics Data")
            st.dataframe(demographics)
            st.write(f"Shape: {demographics.shape}")
        elif sub_category == "Dietary Intake" and not diet_df.empty:
            st.header("Dietary Intake Data")
            st.dataframe(diet_df)
            st.write(f"Shape: {diet_df.shape}")
        elif sub_category == "Examination" and not examination_df.empty:
            st.header("Examination Data")
            st.dataframe(examination_df)
            st.write(f"Shape: {examination_df.shape}")
        elif sub_category == "Laboratory Results" and not labs_df.empty:
            st.header("Laboratory Results")
            st.dataframe(labs_df)
            st.write(f"Shape: {labs_df.shape}")
        elif sub_category == "Questionnaire Data" and not questionnaire_df.empty:
            st.header("Questionnaire Data")
            st.dataframe(questionnaire_df)
            st.write(f"Shape: {questionnaire_df.shape}")
        elif sub_category == "Medication Data" and not medication_df.empty:
            st.header("Medication Data")
            st.dataframe(medication_df)
            st.write(f"Shape: {medication_df.shape}")
        else:
            st.warning(f"{sub_category} data is not available or is empty.")

    else:
        st.error("No data loaded. Please check CSV files and DataProcessor methods.")



if section == "Cluster Profiles":
    clustering_page(merged_df)
   

# GRAPHS AND TRENDS


if section=="Graphs & Trends":

    bins = [0, 2, 12, 18, 35, 50, 65, 80, 120]
    labels = ["Infant (0-2)", "Child (3-12)", "Teen (13-18)", "Young Adult (19-35)",
              "Middle Age (36-50)", "Older Adult (51-65)", "Senior (66-80)", "Elderly (81+)"]

    demographics["Age Group"] = pd.cut(demographics["Age"], bins=bins, labels=labels, right=False)
    

    st.title("Graphs and Trends")

    # AGE & GENDER DISTRIBUTION 
    st.subheader("Age & Gender Distribution")

    age_distribution = demographics["Age Group"].value_counts()
    st.write("Age group distribution summary:")
    st.write(age_distribution)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=age_distribution.index, y=age_distribution.values, palette="magma")
    plt.xticks(rotation=45)
    plt.xlabel("Age Groups")
    plt.ylabel("Number of Individuals")
    plt.title("Age Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig1)
    plt.close(fig1) 

    st.markdown(
    """
    The overall NHANES sample analyzed comprises approximately 10,175 unique participants after data processing.
    In the age distribution plot above, the largest representations are found in the **Child (3–12 years)** and **Young Adult (19–35 years)** categories.
    This indicates a strong focus on these demographic segments within the survey, which is crucial for understanding health trends across different life stages.

    The gender distribution is nearly balanced, consisting of **50.8% females** and **49.2% males**.
    This balanced representation helps ensure that the findings derived from this dataset are broadly generalizable across genders.
    """
    )

    gender_distribution = demographics["RIAGENDR"].value_counts()
    st.write("Gender distribution summary:")
    st.write(gender_distribution)


    # SPECIAL DIETS PREVALENCE 
    st.subheader("Prevalence of Special Diets")

    diet_columns_list = [
        "Weight loss or low calorie diet", "Low fat/Low cholesterol diet",
        "Low salt/Low sodium diet", "Sugar free/Low sugar diet", "Low fiber diet",
        "High fiber diet", "Diabetic diet", "Weight gain/Muscle building diet",
        "Low carbohydrate diet", "High protein diet", "Renal/Kidney diet", "Other special diet"
    ]

    existing_diet_columns = [col for col in diet_columns_list if col in diet_df.columns]
    diet_counts = diet_df[existing_diet_columns].apply(lambda col: (col == "Yes").sum())
    diet_counts = diet_counts[diet_counts > 0] 

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=diet_counts.index, y=diet_counts.values, palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Diet Type")
    plt.ylabel("Count")
    plt.title("Prevalence of Special Diets")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Discussion for Special Diets Prevalence
    st.markdown(
    """
    The "Prevalence of Special Diets" graph provides insights into the dietary habits reported by participants.
    The **"Weight loss or low-calorie diet" is by far the most prevalent, reported by over 500 individuals**.
    This highlights a significant public interest in weight management, which could be driven by health
    recommendations, personal goals, or existing conditions like obesity.

    Following this, the **"Diabetic diet"** and **"Low fat/Low cholesterol diet"** are the next most common. The high prevalence of these specific diets
    strongly suggests that a substantial portion of the population is actively managing their diet due
    to existing health conditions or as a preventive measure.

    Other diets—such as **"Low salt**, **Low sodium diet"** and **"Low-Carbohydrate"**—are also present but with lower counts.
    The overall pattern indicates that dietary modifications are a common strategy among the surveyed
    population, often tailored to specific health needs.
    """
    )

    disease_vars_list = [
        "Ever have Diabetes", "Ever have Prediabetes", "Ever have DiabetesRisk",
        "Ever have Hypertension", "Ever have HighChol", "Ever have Asthma", "Ever have Overweight",
        "Ever have Celiac", "Ever have Cancer", "Ever have Anemia", "Ever have Psoriasis",
        "Ever have Arthritis", "Ever have Gout", "Ever have Congestive Heart Failure",
        "Ever have Coronary Heart Disease", "Ever have Angina", "Ever have Heart Attack",
        "Ever have Stroke", "Ever have Thyroid Problem",
        "Ever have Liver Condition", "Ever have HepB" 
    ]
    existing_disease_columns = [col for col in disease_vars_list if col in questionnaire_df.columns]

    prevalence = questionnaire_df[existing_disease_columns].apply(lambda col: (col == "Yes").mean() * 100)
    prevalence = prevalence.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8)) 
    sns.barplot(x=prevalence.values, y=prevalence.index, palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    plt.xlabel("Prevalence (%)")
    plt.title("Prevalence of Reported Diseases")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Discussion for Prevalence of Reported Diseases
    st.markdown(
    """
    The "Prevalence of Reported Diseases" graph provides a snapshot of the health landscape within the NHANES dataset.
    The most prevalent conditions reported are **"Ever have Hypertension"** and **"Ever have (been told you are) Overweight"**, both exceeding 20% prevalence.
    This highlights the significant public health burden of these conditions, which are often interconnected and major risk factors for other chronic diseases.

    **"Ever have HighChol(High Cholesterol)"** also shows a high prevalence, indicating a widespread issue with lipid management.
    Other frequently reported conditions include **Asthma** and **Arthritis**.
    The **"Ever have Diabetes Risk"**,**"Ever have Diabetes"** and **"Ever have Prediabetes"**  prevalence shows the ongoing challenge of diabetes and prediabetes in the population.

    Conditions such as Hepatitis B and Celiac are among the least reported in this sample.
    In contrast, the high incidence of chronic conditions like hypertension, obesity, and high cholesterol, conditions which are often driven by lifestyle factors, aligns strongly with the motivation of this study, emphasizing the need for data-driven approaches to understand and mitigate these health challenges.
    """
    )

    #  Disease-Diet Co-occurrence Heatmap 
    st.subheader("Disease-Diet Co-occurrence")

    disease_binary_df = questionnaire_df[["SEQN"] + existing_disease_columns].copy()
    for col in existing_disease_columns:
        disease_binary_df[col] = (disease_binary_df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    disease_binary_df["no_disease"] = (disease_binary_df[existing_disease_columns].sum(axis=1) == 0).astype(int)
    all_disease_cols_for_heatmap = existing_disease_columns + ["no_disease"]

    diet_binary_df = diet_df[["SEQN"] + existing_diet_columns].copy()
    for col in existing_diet_columns:
        diet_binary_df[col] = (diet_binary_df[col].astype(str).str.strip().str.lower() == "yes").astype(int)


    merged_disease_diet_df = pd.merge(disease_binary_df, diet_binary_df, on="SEQN", how="inner")

    if merged_disease_diet_df.empty:
        st.warning("No overlapping data between disease and diet records. Cannot compute correlation.")
    else:

        disease_data_for_corr = merged_disease_diet_df[all_disease_cols_for_heatmap]
        diet_data_for_corr = merged_disease_diet_df[existing_diet_columns]

        correlation_matrix = disease_data_for_corr.T.dot(diet_data_for_corr)

        # Percent of people with disease X who follow diet Y
        disease_totals = disease_data_for_corr.sum(axis=0).replace(0, np.nan)
        correlation_percent = correlation_matrix.div(disease_totals, axis=0) * 100

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_percent, annot=True, fmt=".1f", cmap="magma", cbar_kws={'label': '% of Patients'})
        plt.xlabel("Diet Type")
        plt.ylabel("Reported Disease")
        plt.title("Percent of People with Each Disease Following Each Diet")
        st.pyplot(fig)
        plt.close(fig) 

        st.markdown(
            """
            The *"Percent of People with Each Disease Following Each Diet"* heatmap offers insight into how different health conditions correlate with the adoption of specific dietary regimens.

            As expected, the **weight-loss or low-calorie diet** is the most commonly followed—not only by those with **prediabetes (13.3%)**, at **risk for diabetes (12.5%)**, or who are **overweight (13.8%)**, but also among individuals with a wide range of conditions. These include **heart attacks, strokes, thyroid and liver disorders**, as well as **genetically linked diseases** like arthritis and psoriasis. **2.2% of individuals with no reported disease** also adhere to this diet.

            The **diabetic diet** is also widely followed, particularly among those with **diabetes (18.4%)**, which is expected given its role in disease management.

            Similarly, **low-fat/low-cholesterol** and **low-salt diets** are more commonly adopted by individuals with **cardiovascular conditions**, reflecting standard medical guidance.

            Overall, the heatmap highlights that special diets are primarily adopted in response to specific health concerns. This emphasizes the **reactive nature of dietary changes** and the critical role of nutrition in disease management.
            """
        )


    #  Average Nutrient Intake Heatmap 
    st.subheader("Average Nutrient Intake for People with Each Disease")

    nutrient_columns_list = [
        "Energy (kcal)", "Protein (gm)", "Carbohydrate (gm)", "Total sugars (gm)", "Dietary fiber (gm)", "Total fat (gm)",
        "Total saturated fatty acids (gm)", "Total monounsaturated fatty acids (gm)", "Total polyunsaturated fatty acids (gm)",
        "Vitamin B6 (mg)", "Folic acid (mcg)",
        "Vitamin B12 (mcg)", "Vitamin C (mg)",  "Calcium (mg)", 
        "Magnesium (mg)", "Iron (mg)", "Zinc (mg)", "Sodium (mg)", "Potassium (mg)", "Selenium (mcg)", "Caffeine (mg)", "Alcohol (gm)"
    ]
    
    disease_vars_list2 = [
        "Ever have Diabetes", "Ever have Prediabetes", "Ever have DiabetesRisk",
        "Ever have Hypertension", "Ever have HighChol",  "Ever have Overweight",
        "Ever have Anemia",  "Ever have Gout", "Ever have Congestive Heart Failure",
        "Ever have Coronary Heart Disease", "Ever have Angina", "Ever have Heart Attack",
        "Ever have Stroke", "Ever have Thyroid Problem",
        "Ever have Liver Condition"
    ]

    existing_nutrient_columns = [col for col in nutrient_columns_list if col in diet_df.columns]
    nutrient_data_for_merge = diet_df[["SEQN"] + existing_nutrient_columns].copy()

    merged_nutrient_disease_df = pd.merge(disease_binary_df, nutrient_data_for_merge, on="SEQN", how="inner")

    if merged_nutrient_disease_df.empty:
        st.warning("No overlapping data between nutrient and disease records. Cannot compute average nutrient intake heatmap.")
    else:
        nutrient_means = {}
    
    diseases_for_nutrient_heatmap_rows = disease_vars_list2
    if "no_disease" in merged_nutrient_disease_df.columns:
        diseases_for_nutrient_heatmap_rows.append("no_disease")


    for disease in diseases_for_nutrient_heatmap_rows:
        group = merged_nutrient_disease_df[merged_nutrient_disease_df[disease] == 1].copy() 
        if not group.empty:
            nutrient_means[disease] = group[existing_nutrient_columns].mean()
        else:
            nutrient_means[disease] = pd.Series(np.nan, index=existing_nutrient_columns) 

    nutrient_df_for_heatmap = pd.DataFrame(nutrient_means).T

    # --- Apply standard Z-score normalization (Overall Population) ---
    nutrient_df_normalized = (nutrient_df_for_heatmap - nutrient_df_for_heatmap.mean()) / nutrient_df_for_heatmap.std()
    cbar_label = 'Z-score (vs. Overall Population)'
    heatmap_title = "Average Nutrient Intake Among People with Each Disease (Z-score Normalized vs. Overall Population)"

    fig, ax = plt.subplots(figsize=(18, 12))
    nutrient_df_normalized_plot = nutrient_df_normalized.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    
    if not nutrient_df_normalized_plot.empty:
        sns.heatmap(nutrient_df_normalized_plot, cmap="coolwarm", annot=True, linewidths=0.5, cbar_kws={'label': cbar_label})
        plt.xlabel("Nutrient")
        plt.ylabel("Reported Disease")
        plt.title(heatmap_title)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Normalized nutrient data is empty after processing. Cannot generate heatmap.")
        plt.close(fig) 

    st.markdown(
    """
    ### Nutrient Intake and Disease Association (Z-score Normalized Heatmap)

    The "Average Nutrient Intake for People with Each Disease" heatmap—normalized using Z-scores against the general population—provides insight into how dietary patterns differ among individuals with specific health conditions. In this visualization, red shades represent a positive Z-score, indicating nutrient intake above the population average for that disease group. Blue shades represent negative Z-scores, showing below-average intake. Neutral shades near zero reflect values close to the general average.
    One of the most striking observations is the high alcohol intake among individuals reporting liver conditions. Their deep red Z-score for "Alcohol (gm)" aligns with well-established links between alcohol consumption and liver disease. 

    In terms of diabetes, individuals diagnosed with the condition appear to follow a more controlled dietary pattern. Their Z-scores for carbohydrates, sugars, and various fats are generally closer to zero, possibly indicating adherence to dietary guidelines aimed at managing diabetes. On the other hand, individuals who report being at risk of diabetes or having prediabetes tend to show significantly higher Z-scores in energy, carbohydrate, sugar, and fat intake. This may reflect dietary behaviors contributing to the early development of the disease.

    A noteworthy pattern emerges in cardiovascular conditions such as congestive heart failure, coronary heart disease, angina, and heart attacks. People with these conditions show elevated Z-scores for caffeine intake. While this doesn’t establish causation, it highlights an area worth deeper investigation.

    Among those diagnosed with anemia, iron and folic acid intake is above average. These elevated Z-scores likely reflect supplementation or dietary advice aimed at correcting nutrient deficiencies commonly associated with anemia.

    People who report being overweight or having high cholesterol also tend to consume higher amounts of total fat and saturated fat, as reflected by their elevated Z-scores. These findings align with dietary risk factors for metabolic diseases and cardiovascular complications.

    Overall, the heatmap serves as a valuable tool for identifying patterns of nutrient intake that correlate with specific health conditions. It not only highlights dietary risk factors but also reveals potential evidence of dietary interventions or supplementation, offering useful insights for both public health policy and personalized nutrition strategies.
    """
)
     # Perform Kruskal-Wallis tests and display results
    kruskal_results = []
    for nutrient in existing_nutrient_columns:
        groups = []
        for disease_status in all_disease_cols_for_heatmap:
            group_data = merged_nutrient_disease_df[merged_nutrient_disease_df[disease_status] == 1][nutrient].dropna()
            if not group_data.empty:
                groups.append(group_data)

        if len(groups) >= 2:
            try:
                h_statistic, p_value = stats.kruskal(*groups)
                kruskal_results.append({
                    'Nutrient': nutrient,
                    'H-statistic': h_statistic,
                    'p-value': p_value
                })
            except ValueError as e:
                kruskal_results.append({
                    'Nutrient': nutrient,
                    'H-statistic': np.nan,
                    'p-value': f"Error: {e}"
                })
        else:
            kruskal_results.append({
                'Nutrient': nutrient,
                'H-statistic': np.nan,
                'p-value': "Not enough groups for test"
            })

    kruskal_df = pd.DataFrame(kruskal_results)
    formatted_p_val = "< 0.0000000001" if p_value < 1e-10 else f"{p_value:.10f}"

    kruskal_df['formatted p-value'] = kruskal_df['p-value'].apply(
    lambda p: "< 0.0000000001" if p < 1e-10 else f"{p:.10f}"
)

    st.dataframe(
        kruskal_df.style.format({
            'H-statistic': '{:.2f}',
            'formatted p-value': '{}'
        }, na_rep="-")
    )

    st.markdown(
        """
        ### Interpreting the Kruskal-Wallis Test Results:

        * A **small p-value** (typically < 0.05) indicates a statistically significant difference in **median intake** of that nutrient across some disease groups.
        * A **large p-value** (≥ 0.05) suggests **no significant difference**, meaning nutrient intake is similar across groups.
        
        These tests **quantify** the differences visualized in the heatmap and help identify nutrients with strong, disease-specific consumption patterns.
        """
    )

    # ---- MOST COMMON MEDICATIONS ----
    st.subheader("Top 10 Most Common Medications")

    med_counts = medication_df["Medication Name"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=med_counts.index, y=med_counts.values, palette="magma")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Medication Name")
    plt.ylabel("Count")
    plt.title("Top 10 Most Common Medications")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Discussion for Most Common Medications
    st.markdown(
    """
    The "Top 10 Most Common Medications" graph offers a direct reflection of the most prevalent health conditions being managed within the surveyed population.
    **Lisinopril** is indicated as the most commonly used medication, followed closely by **Metformin** and **Levothyroxine**.

    A significant portion of the top medications—including **Lisinopril, Simvastatin, Atorvastatin, Amlodipine, Metoprolol, and Hydrochlorothiazide**—are primarily cardiovascular agents.
    These are prescribed for conditions such as hypertension, high cholesterol, and various heart conditions.
    This suggests a substantial burden of cardiovascular disease and its risk factors in the population.

    **Metformin** is a key medication for Type 2 Diabetes management.
    Its high ranking indicates a considerable number of individuals diagnosed with and managing diabetes.

    **Levothyroxine** is commonly used for hypothyroidism (underactive thyroid), and **Omeprazole** is used for gastrointestinal issues like GERD or ulcers.
    Their presence in the top 10 points to other common chronic health issues beyond cardiovascular and metabolic diseases.

    """
    )


 #  SOCIOECONOMIC FACTORS ANALYSIS 
    st.header("Socioeconomic Factors and Their Relation to Health & Diet")

    edu_col_name = "Education Level (Adults 20+)"
    pir_col_name = "Income_to_Poverty_Category"
    age_col_name = "Age"

    edu_order = ["Less than 9th grade", "9-11th grade (No diploma)",
                 "High school graduate/GED", "Some college or AA degree",
                 "College graduate or above"]
    pir_order = ["PIR <1.0 (Below Poverty)", "PIR 1.0-1.99 (Low Income)",
                 "PIR 2.0-3.99 (Middle Income)", "PIR >=4.0 (Higher Income)"]

    #  Distribution of SES Factors 
    st.subheader("Distribution of Socioeconomic Factors")
    df_adults_for_edu_dist = merged_df[merged_df[age_col_name] >= 20].copy()

    if not df_adults_for_edu_dist.empty and edu_col_name in df_adults_for_edu_dist.columns and df_adults_for_edu_dist[edu_col_name].notna().any():
        st.markdown(f"#### {edu_col_name} Distribution")

        edu_counts_dist = df_adults_for_edu_dist[edu_col_name].value_counts().reindex(edu_order).fillna(0)
        fig_edu_dist, ax_edu_dist = plt.subplots(figsize=(10, 6))
        sns.barplot(x=edu_counts_dist.index, y=edu_counts_dist.values, ax=ax_edu_dist, palette="viridis", order=edu_order)
        for container in ax_edu_dist.containers:
            ax_edu_dist.bar_label(container, fmt='%.0f')
        ax_edu_dist.set_ylabel("Number of Participants (Unweighted)"); ax_edu_dist.set_title(f"{edu_col_name}")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_edu_dist); plt.close(fig_edu_dist)

        # Discussion for Education Level Distribution
        st.markdown(
        """
        The "Education Level (Adults 20+)" graph illustrates the distribution of educational attainment among adult participants in the NHANES dataset.
        The largest group reported **"Some college or AA degree" (approximately 1,750 participants)**, closely followed by **"College graduate or above" (approximately 1,450 participants)**.
        This indicates a significant proportion of the adult population in the sample has pursued higher education.

        "High school graduate/GED" also represents a substantial segment (around 1,300 participants).
        In contrast, **"Less than 9th grade" education was the least common category (approximately 450 participants)**.
        This distribution suggests generally higher educational attainment within the surveyed adult population, which can have implications for health literacy, access to resources, and overall health outcomes.
        """
        )
    else:
        st.write(f"{edu_col_name} or {age_col_name} data not available")

    if pir_col_name in merged_df.columns:
        st.markdown(f"#### {pir_col_name} Distribution")
        pir_counts_dist = merged_df[pir_col_name].value_counts().reindex(pir_order).fillna(0)
        fig_pir_dist, ax_pir_dist = plt.subplots(figsize=(10, 6))
        sns.barplot(x=pir_counts_dist.index, y=pir_counts_dist.values, ax=ax_pir_dist, palette="magma", order=pir_order)
        for container in ax_pir_dist.containers:
            ax_pir_dist.bar_label(container, fmt='%.0f')
        ax_pir_dist.set_ylabel("Number of Participants (Unweighted)"); ax_pir_dist.set_title(f"{pir_col_name}")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_pir_dist); plt.close(fig_pir_dist)

        # Discussion for Income Category Distribution
        st.markdown(
        """
        The "Income to Poverty Category" graph displays the distribution of participants across different income-to-poverty ratio (PIR) categories.
        The **"PIR < 1.0 (Below Poverty)"** category is the most common, encompassing over 2,500 participants.
        This indicates that a significant portion of the surveyed population faces economic hardship, living below the poverty line.

        This is followed by **"PIR 1.0–1.99 (Low Income)"**  and **"PIR 2.0–3.99 (Middle Income)"**.
        The **"PIR ≥ 4.0 (Higher Income)"** group has the fewest participants.
        This distribution highlights prevalent economic vulnerability within the NHANES sample, which is a crucial social determinant of health, affecting access to nutritious food, quality healthcare, and safe living environments.
        """
        )
    else:
        st.write(f"{pir_col_name} data not available in merged_df.")
    st.markdown("---")

    #  Relationship between Education and PIR
    st.subheader("Relationship between Education Level and Income-to-Poverty Ratio (Adults 20+)")
    if edu_col_name in merged_df.columns and pir_col_name in merged_df.columns and age_col_name in merged_df.columns:
        df_adults_edu_pir = merged_df[merged_df[age_col_name] >= 20].copy()

        df_adults_edu_pir[edu_col_name] = pd.Categorical(df_adults_edu_pir[edu_col_name], categories=edu_order, ordered=True)
        df_adults_edu_pir[pir_col_name] = pd.Categorical(df_adults_edu_pir[pir_col_name], categories=pir_order, ordered=True)

        if not df_adults_edu_pir.empty and df_adults_edu_pir[edu_col_name].notna().any() and df_adults_edu_pir[pir_col_name].notna().any():
            contingency_edu_pir_counts = pd.crosstab(df_adults_edu_pir[edu_col_name], df_adults_edu_pir[pir_col_name], dropna=False)
            edu_pir_crosstab_percent = contingency_edu_pir_counts.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)
            edu_pir_crosstab_percent = edu_pir_crosstab_percent.reindex(index=edu_order, columns=pir_order).fillna(0)

            fig_edu_pir, ax_edu_pir = plt.subplots(figsize=(12, 8))
            sns.heatmap(edu_pir_crosstab_percent, annot=True, fmt=".1f", cmap="BuPu", ax=ax_edu_pir, cbar_kws={'label': '% within Education Level'})
            ax_edu_pir.set_title("Distribution of PIR Categories within each Education Level (Adults 20+)")
            plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout(); st.pyplot(fig_edu_pir); plt.close(fig_edu_pir)

            # Discussion for Education Level with Income Heatmap
            st.markdown(
            """
            The "Distribution of PIR Categories within each Education Level (Adults 20+)" heatmap clearly illustrates the strong interrelationship between educational attainment and income-to-poverty ratio.

            - For individuals with **"Less than 9th grade" education**, a disproportionately large percentage falls into **"PIR < 1.0 (Below Poverty)" (47.6%)** and **"PIR 1.0–1.99 (Low Income)" (35.7%)**.
              This indicates a high concentration of economic vulnerability among those with the lowest educational attainment.

            - Conversely, for participants with **"College graduate or above"** education, the majority (a striking **57.1%**) fall into the **"PIR ≥ 4.0 (Higher Income)"** category.
              This highlights a strong positive correlation, where higher education is associated with significantly higher income levels.

            This heatmap visually confirms a well-established socioeconomic trend: educational attainment is a powerful predictor of economic status.
            Understanding this relationship is crucial because both education and income are fundamental social determinants that profoundly influence health outcomes.
            """
            )

            if contingency_edu_pir_counts.shape[0] >=2 and contingency_edu_pir_counts.shape[1] >=2 and contingency_edu_pir_counts.sum().sum() > 0:
                try:
                    chi2_edu_pir, p_edu_pir, _, _ = stats.chi2_contingency(contingency_edu_pir_counts)
                    st.write(f"Chi-squared test for association between Education and PIR: χ²={chi2_edu_pir:.2f}, p-value={formatted_p_val} {'(Statistically Significant)' if p_edu_pir < 0.05 else ''}")
                except Exception as e_chi_edu_pir: st.write(f"Could not compute Chi-squared for Education vs PIR: {e_chi_edu_pir}")
        else: st.write("Not enough data for Education vs. PIR analysis (after filtering for adults 20+ and ensuring both variables are present).")
    else: st.write("Education, PIR, or Age data not available for joint analysis.")
    st.markdown("---")

    # Disease Prevalence by Individual and Joint SES Factors 
    st.subheader("Disease Prevalence by Socioeconomic Factors")

    diseases_to_analyze_ses = {
        "Ever have Diabetes": "Diabetes Prevalence (%)",
        "Ever have Hypertension": "Hypertension Prevalence (%)"
    }


    # Joint SES and Disease Prevalence
    if edu_col_name in merged_df.columns and pir_col_name in merged_df.columns and age_col_name in merged_df.columns:
        df_adults_joint_disease = merged_df[merged_df[age_col_name] >= 20].copy()
        df_adults_joint_disease[edu_col_name] = pd.Categorical(df_adults_joint_disease[edu_col_name], categories=edu_order, ordered=True)
        df_adults_joint_disease[pir_col_name] = pd.Categorical(df_adults_joint_disease[pir_col_name], categories=pir_order, ordered=True)

        if not df_adults_joint_disease.empty:
            for disease_col, disease_label in diseases_to_analyze_ses.items():
                if disease_col in df_adults_joint_disease.columns and df_adults_joint_disease[disease_col].notna().any():
                    df_adults_joint_disease[disease_col] = df_adults_joint_disease[disease_col].astype(str).str.lower()
                    if 'yes' in df_adults_joint_disease[disease_col].unique():
                        st.markdown(f"#### {disease_label} by Education and PIR Category")
                        try:
                            joint_prev_df = df_adults_joint_disease.groupby([edu_col_name, pir_col_name], observed=False)[disease_col].apply(lambda x: (x == 'yes').mean() * 100 if len(x) > 0 else np.nan).unstack(fill_value=np.nan)
                            joint_prev_df = joint_prev_df.reindex(index=edu_order, columns=pir_order)

                            if not joint_prev_df.empty:
                                fig_joint_disease, ax_joint_disease = plt.subplots(figsize=(12, 8))
                                sns.heatmap(joint_prev_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax_joint_disease, linewidths=.5, cbar_kws={'label': 'Prevalence (%)'})
                                ax_joint_disease.set_title(f"{disease_label} by Education and PIR"); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout(); st.pyplot(fig_joint_disease); plt.close(fig_joint_disease)

                                # Discussion for Joint SES and Disease Prevalence
                                if disease_col == "Ever have Diabetes":
                                    st.markdown(
                                    """
                                    Individuals with less than a 9th-grade education report the highest diabetes prevalence across all income groups, ranging from approximately 24% to 27%. 
                                    Even among those in the highest income bracket (PIR ≥ 4.0), diabetes rates remain elevated for this education group, indicating that income alone does not eliminate risk when educational attainment is limited.
                                    These findings underscore the complex relationship between socioeconomic factors and health outcomes.
                                    While both income and education contribute to disease risk, the data suggest that education may play a particularly pivotal role, potentially through its influence on health behavior, preventative care, and long-term disease management.
                                    """
                                    )
                                elif disease_col == "Ever have Hypertension":
                                    st.markdown(
                                    """
                                    Unlike diabetes, the relationship between hypertension, income, and education is less consistent. 
                                    While some groups with lower education and lower income show higher rates of hypertension, other groups do not follow this pattern. For instance, individuals with less than a 9th-grade education and higher income (PIR ≥ 4.0) show much lower rates (28.6%), while some middle-income groups have unexpectedly high rates.

                                    Overall, the chart suggests that education and income may influence hypertension risk, but not in a straightforward way. Other factors, such as age, lifestyle, or access to care, may be affecting these patterns, making the relationship more complex.
                                                                            """
                                    )
                            else:
                                st.write(f"Not enough data.")
                        except Exception as e_joint_disease: st.warning(f"Could not generate joint SES analysis.")
                    else: st.write(f"No 'Yes' responses for '{disease_col}' to create joint SES heatmap.")
                    try:
                        contingency_table = pd.crosstab(df_adults_joint_disease[edu_col_name], 
                                                        df_adults_joint_disease[pir_col_name], 
                                                        values=(df_adults_joint_disease[disease_col] == 'yes').astype(int),
                                                        aggfunc='sum').reindex(index=edu_order, columns=pir_order).fillna(0)

                        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2 and contingency_table.values.sum() > 0:
                            chi2_val, p_val, _, _ = stats.chi2_contingency(contingency_table)
                            
                            st.markdown(f"**Chi-squared Test (Education × PIR → {disease_label})**: χ² = {chi2_val:.2f}, p = {formatted_p_val} {'(**Statistically significant**)' if p_val < 0.05 else '(Not significant)'}")
                        else:
                            st.write(f"Not enough data for Chi-squared test on {disease_label} by Education and PIR.")
                    except Exception as e:
                        st.write(f"Could not compute Chi-squared test for {disease_label}: {e}")
                    st.markdown("---")




    st.header("Overall Discussion and Conclusion from Exploratory Data Analysis")
    st.markdown(
    """
    This exploratory data analysis of the NHANES dataset offers context of the surveyed population, highlighting critical connections between various factors.

    The analysis consistently reveals a substantial burden of **chronic diseases**  which are prevalent public health concerns. The widespread adoption of special diets, such as those for weight loss or diabetes management, underscores a **reactive approach** to health, where dietary changes are often initiated in response to existing conditions. This highlights the pivotal role of nutrition in disease management, yet also points to opportunities for more proactive, preventive interventions.

    Heatmaps showed dietary patterns linked to various health conditions. For instance, the strong association between **liver conditions and high alcohol consumption**,  aligns with established medical knowledge. Conversely, individuals at risk of diabetes exhibited higher energy, carbohydrate, and fat intake, suggesting dietary behaviors that contribute to disease progression, unlike diagnosed diabetics who appear to follow more controlled diets.

    Beyond dietary specifics, **socioeconomic factors emerged as fundamental determinants of health**. A clear correlation between higher educational attainment and increased income levels was observed. Significantly, lower educational attainment was associated with a higher prevalence of diabetes, even across different income brackets, underscoring education's profound influence on health behaviors and access to preventive care.

    This exploratory analysis, while revealing compelling patterns, is subject to certain **limitations**. Being based on a **cross-sectional dataset**, it provides insights at a single point in time, meaning **causal relationships cannot be directly inferred**. Observed correlations indicate associations that warrant further investigation, but they do not establish cause and effect. Additionally, the reliance on **self-reported data** for dietary intake, medication use, and disease history may introduce **recall bias**, potentially affecting the accuracy of some findings.

    Despite these limitations, the insights derived are valuable. They illuminate key public health challenges, identify demographic segments with particular dietary and health profiles, and highlight the influence of socioeconomic factors on well-being. This data-driven understanding is crucial for informing targeted public health strategies and adopting a more nuanced approach to disease prevention and management.
    """
)

   

if section == "Predictive Modeling":
    predictive_page(demographics, diet_df, examination_df, labs_df, questionnaire_df, medication_df)
