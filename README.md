# Interactive Patient Data Analysis with NHANES

This repository contains a Streamlit web application designed for the interactive exploration and analysis of health data. 

## Data Source

The analyses presented here use data from the National Health and Nutrition Examination Survey (NHANES), a program by the U.S. Centers for Disease Control and Prevention (CDC). The cycle used is this one - [NHANES on Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey/data). The required files need to be added in a "data" folder at the root of the project.

demographic.csv

diet.csv

examination.csv

labs.csv

medications.csv

questionnaire.csv

## Installation Guide


### 1. Prerequisites

* **Python 3.12**
* **Git** 
* **pip** 

### 2. Clone the Repository

Open your terminal or command prompt and clone this repository. This will include the source code.

```bash
git clone <your-repository-url>
cd <repository-directory>
```



### 3. Set Up a Virtual Environment (Optional)

It is recommended to create a virtual environment to avoid conflicts with other Python projects.

* **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

* **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 4. Install Required Libraries

This project uses several Python libraries.

Create a file named `requirements.txt` in the root of your project directory and add the following lines:

```
streamlit
pandas
seaborn
matplotlib
scipy
numpy
statsmodels
gower
scikit-learn
kmedoids
shap
xgboost
```

Install all the required packages:

```bash
pip install -r requirements.txt
```



## 5. How to Run the Application

The following directory structure is expected:
```
.
├── data/
│   ├── demographic.csv
│   ├── diet.csv
│   ├── examination.csv
│   ├── labs.csv
│   ├── medications.csv
│   └── questionnaire.csv
├── app.py
├── clustering.py
├── predictive.py
├── requirements.txt
└── README.md
```

Once you have completed the installation, you can run the Streamlit application.

Ensure you are in the project's root directory and your virtual environment is activated. Then, execute the following command:

```bash
streamlit run app.py
```

The application should now open in a new tab in your default web browser.
