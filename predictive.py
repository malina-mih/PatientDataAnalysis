import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score

# Helper function
def plot_roc_pr_curves(y_true, y_pred_proba, model_name, disease_name, ax_roc, ax_pr):
    """Plots ROC and Precision-Recall curves"""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve for {disease_name}')
    ax_roc.legend()
    ax_roc.grid(True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    ax_pr.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Precision-Recall Curve for {disease_name}')
    ax_pr.legend()
    ax_pr.grid(True)

#  feature sets for each disease 

DIABETES_FEATURES = {
    "target": "Ever have Diabetes",
    "demographics": ['SEQN', 'Age', 'RIAGENDR', 'Education Level (Adults 20+)', 'PIR_Ratio'],
    "diet": ["SEQN", "Energy (kcal)", "Alcohol (gm)"],
    "examination": ['SEQN', 'Body Mass Index (kg/m²)', 'Systolic Blood Pressure (1st reading) mm Hg', 'Diastolic Blood Pressure (1st reading) mm Hg'], 
    "labs": ['SEQN', 'Direct HDL-Cholesterol (mg/dL)', 'Total cholesterol (mg/dL)', 'Triglyceride (mg/dL)', 'Hemoglobin (g/dL)', 'Insulin (uU/mL)'],
    "questionnaire": ['SEQN', 'Vigorous Recreational Activity (Yes/No)', 'Minutes Sedentary Activity per Day', 'Hours Sleep Weekdays', 'Smoking Status Now']
}

HYPERTENSION_FEATURES = {
    "target": "Ever have Hypertension",
    "demographics": ['SEQN', 'Age', 'RIAGENDR', 'Education Level (Adults 20+)', 'PIR_Ratio'],
    "diet": ["SEQN", "Energy (kcal)", "Alcohol (gm)", "Sodium (mg)"], 
    "examination": ['SEQN', 'Body Mass Index (kg/m²)'], 
    "labs": ['SEQN', 'Direct HDL-Cholesterol (mg/dL)', 'Total cholesterol (mg/dL)', 'Triglyceride (mg/dL)', 'Hemoglobin (g/dL)', 'Insulin (uU/mL)'],
    "questionnaire": ['SEQN', 'Vigorous Recreational Activity (Yes/No)', 'Minutes Sedentary Activity per Day', 'Hours Sleep Weekdays', 'Smoking Status Now']
}


# --- Data Preparation for Prediction (Unified for Diabetes and Hypertension) ---
@st.cache_data(ttl=3600)
def prepare_features_for_model(demographics_data, diet_data, examination_data, labs_data, questionnaire_data, disease_to_predict):
    """
    Prepares and engineers features from raw NHANES dataframes for a specific disease prediction task.

    This function handles:
    - Selection of relevant features from different data sources based on the specified disease.
    - Engineering new features 
    - Merging dataframes.
    - Defining preprocessing pipelines for numerical and categorical features.
    - Explicitly excluding features that would cause data leakage.

    Args:
        dataframes
        disease_to_predict: either "Diabetes" or "Hypertension".

    Returns:
        tuple: (X_data, y_data, preprocessor_pipeline, numerical_features, categorical_features)
               Returns None if data preparation fails.
    """
    feature_sets = None
    if disease_to_predict == "Diabetes":
        feature_sets = DIABETES_FEATURES
    elif disease_to_predict == "Hypertension":
        feature_sets = HYPERTENSION_FEATURES
    else:
        return None, None, None, None, None

    target_col_name_original = feature_sets["target"]

    # Feature Selection & Engineering 

    # Demographics: Age, Gender, Education, Income
    dem_cols_initial = feature_sets["demographics"]
    df_demo_select = demographics_data[dem_cols_initial].copy()

    # Engineer 'Has College Degree' from education level
    education_col_original = "Education Level (Adults 20+)"
    has_college_cond = (df_demo_select[education_col_original].astype(str).str.lower() == 'college graduate or above')
    df_demo_select['Has College Degree'] = np.nan
    df_demo_select.loc[df_demo_select[education_col_original].notna(), 'Has College Degree'] = 0
    df_demo_select.loc[has_college_cond, 'Has College Degree'] = 1

    # Diet Features
    diet_feature_cols_input = feature_sets["diet"]
    df_diet_features = diet_data[diet_feature_cols_input].copy()
    for col in [c for c in diet_feature_cols_input if c != 'SEQN']:
        df_diet_features[col] = pd.to_numeric(df_diet_features[col], errors='coerce')

    # Examination Features
    exam_feature_cols = feature_sets["examination"]
    df_exam_select = examination_data[exam_feature_cols].copy()
    for col in [c for c in exam_feature_cols if c != 'SEQN']:
        df_exam_select[col] = pd.to_numeric(df_exam_select[col], errors='coerce')

    # Lab Features
    lab_feature_cols = feature_sets["labs"]
    df_labs_select = labs_data[lab_feature_cols].copy()
    for col in [c for c in lab_feature_cols if c != 'SEQN']:
        df_labs_select[col] = pd.to_numeric(df_labs_select[col], errors='coerce')

    # Questionnaire/Lifestyle Features
    lifestyle_cols_initial = feature_sets["questionnaire"]
    df_lifestyle_select = questionnaire_data[lifestyle_cols_initial].copy()

    # Convert 'Vigorous Recreational Activity (Yes/No)' to binary 
    df_lifestyle_select['Vigorous Recreational Activity (Yes/No)'] = (
        df_lifestyle_select['Vigorous Recreational Activity (Yes/No)'].astype(str).str.lower() == 'yes'
    ).astype(int)

    for col in ['Minutes Sedentary Activity per Day', 'Hours Sleep Weekdays']:
        df_lifestyle_select[col] = pd.to_numeric(df_lifestyle_select[col], errors='coerce')

    # Refactor Smoking Status logic 
    smoking_map = {'every day smoker': 1, 'some days smoker': 1, 'not at all smoker': 0}
    df_lifestyle_select['Current Smoker'] = df_lifestyle_select['Smoking Status Now'].str.lower().map(smoking_map)
    df_lifestyle_select.drop(columns=['Smoking Status Now'], inplace=True)

    # Target Variable Preparation
    if target_col_name_original not in questionnaire_data.columns:
        st.error(f"Target column '{target_col_name_original}' not found in Questionnaire data.")
        return None, None, None, None, None
    df_target = questionnaire_data[['SEQN', target_col_name_original]].copy()
    df_target.dropna(subset=[target_col_name_original], inplace=True)
    df_target['Target'] = (df_target[target_col_name_original].astype(str).str.lower() == 'yes').astype(int)
    if df_target['Target'].nunique() < 2:
        return None, None, None, None, None

    # Merging and Finalizing Data
    df_predict = df_target[['SEQN', 'Target']].copy()
    dfs_to_merge_for_pred = [df_demo_select, df_diet_features, df_exam_select, df_labs_select, df_lifestyle_select]
    for df_feat in dfs_to_merge_for_pred:
        df_predict = pd.merge(df_predict, df_feat, on='SEQN', how='left')

    df_predict = df_predict.set_index('SEQN')

    X_data_intermediate = df_predict.drop(columns=['Target'], errors='ignore')
    X_data = X_data_intermediate.drop(columns=[education_col_original], errors='ignore')
    y_data = df_predict['Target']

    if X_data.empty or y_data.empty:
        st.error(f"Feature set X or target y is empty after merging.")
        return None, None, None, None, None

    # Define Feature Lists for Preprocessing
    numerical_features = X_data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure binary engineered columns are numerical
    binary_as_numerical = ['Has College Degree', 'Current Smoker', 'Vigorous Recreational Activity (Yes/No)']
    for feat in binary_as_numerical:
        if feat in categorical_features:
            categorical_features.remove(feat)
        if feat not in numerical_features:
            numerical_features.append(feat)

    # Ensure gender is categorical
    if 'RIAGENDR' in numerical_features:
        numerical_features.remove('RIAGENDR')
    if 'RIAGENDR' not in categorical_features:
        categorical_features.append('RIAGENDR')

    with st.expander(f"See Detailed Feature Selection & Preprocessing for {disease_to_predict}", expanded=False):
        st.markdown(f"**A. Feature Selection & Engineering:**")
        st.write("- *Demographics:* Age, Gender, Has College Degree, PIR_Ratio")
        st.write(f"- *Diet:* {', '.join(col.replace(' (gm)','').replace(' (kcal)','').replace(' (mg)','') for col in feature_sets['diet'] if col != 'SEQN')}")
        st.write(f"- *Examination:* {', '.join(col.replace(' (kg/m²)', '').replace(' (1st reading) mm Hg','') for col in feature_sets['examination'] if col != 'SEQN')}")
        st.write(f"- *Labs:* {', '.join(col.replace(' (mg/dL)', '').replace(' (g/dL)','').replace(' (uU/mL)','') for col in feature_sets['labs'] if col != 'SEQN')}")
        st.write(f"- *Lifestyle:* {', '.join(col.replace(' (Yes/No)','') for col in feature_sets['questionnaire'] if col not in ['SEQN', 'Smoking Status Now'])}, Current Smoker") # Adjust for engineered smoking col

        st.markdown("**B. Preventing Data Leakage:**")
        st.info(
            "To build a realistic predictive model, certain features that are part of the diagnostic criteria "
            "or are direct results of the condition are intentionally **excluded** from the feature set."
        )
        if disease_to_predict == "Hypertension":
            st.markdown(
                "- **Blood Pressure readings** were **excluded** from the model predicting Hypertension, "
                "as high blood pressure is the definition of the condition. Including it would make the model trivially perfect but useless for prediction."
            )
        if disease_to_predict == "Diabetes":
            st.markdown(
                "- **Glycohemoglobin (A1C)** levels were **excluded**. A1C is a primary diagnostic test for Diabetes, "
                "so including it would directly leak the outcome information to the model."
            )

        st.markdown(f"**C. Target Variable:** '{target_col_name_original}' converted to 0 (No) / 1 (Yes).")
        st.markdown(f"**D. Final Features for Preprocessing:**")
        st.markdown(f"  - **Numerical Features for Scaling:** `{'`, `'.join(sorted(numerical_features))}`")
        st.markdown(f"  - **Categorical Features for Encoding:** `{'`, `'.join(sorted(categorical_features))}`")

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return X_data, y_data, preprocessor_pipeline, numerical_features, categorical_features


def predictive_page(demographics, diet_df, examination_df, labs_df, questionnaire_df, medication_df):
    
    analysis_target = st.sidebar.radio(
        "Select Outcome to Predict:",
        ("Diabetes Risk", "Hypertension Risk"),
        key="prediction_target_select_main_v2" 
    )

    disease_title = "Diabetes" if analysis_target == "Diabetes Risk" else "Hypertension"
    
    st.title(f"Predictive Modeling: {disease_title} Risk")
    st.markdown(f"""
    This section uses participant data to build machine learning models to predict the likelihood of an individual having **{disease_title}**.
    The process involves data preparation, model training, performance evaluation, and feature importance analysis.
    """)
    
    required_dfs = {
        'demographics': demographics, 'diet_df': diet_df, 'examination_df': examination_df,
        'labs_df': labs_df, 'questionnaire_df': questionnaire_df
    }
    for name, df_obj in required_dfs.items():
        if df_obj is None or df_obj.empty:
            st.error(f"Required DataFrame '{name}' is missing or empty. Cannot proceed with {disease_title} prediction.")
            st.stop()

    st.header(f"1. Preparing the Data for {disease_title} Prediction")
    
    with st.spinner(f"Preparing features for {disease_title}..."):
        prep_data_output = prepare_features_for_model(
            demographics, diet_df, examination_df,
            labs_df, questionnaire_df,
            disease_title 
        )

    if prep_data_output is None or any(item is None for item in prep_data_output):
        st.error(f"Data preparation failed for {disease_title}. Check logs or expander above.")
        st.stop()
    
    X, y, preprocessor, numerical_features, categorical_features = prep_data_output

    if X.empty or y.empty or y.nunique() < 2:
        st.error(f"Data preparation for {disease_title} resulted in empty features.")
        st.stop()

    st.success(f"Data preparation for {disease_title} complete! Shape of feature set X: {X.shape}, Target y: {y.shape}")
    target_col = "Ever have Diabetes" if disease_title == "Diabetes" else "Ever have Hypertension"
    st.markdown(f"**Target Variable Distribution ({target_col}):**")
    st.dataframe(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    st.markdown("### Train-Test Split")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        st.write(f"Training set: {X_train.shape[0]} participants, Test set: {X_test.shape[0]} participants.")
    except ValueError as e_split:
        st.error(f"Error during train-test split for {disease_title} (likely insufficient samples): {e_split}")
        st.stop()

    st.header(f"2. Model Training & Evaluation for {disease_title}")
    
    scale_pos_weight_val = 1.0 
    if not y_train.empty and y_train.sum() > 0: 
        scale_pos_weight_val = (y_train == 0).sum() / y_train.sum()
    
    models = {
        "Logistic Regression (L2)": LogisticRegression(penalty='l2', solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000),
        "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=scale_pos_weight_val)
    }

    cv_key = disease_title.lower().replace(" ", "_")
    st.info("Using 10-fold cross-validation for model evaluation.")
    cv_folds = 10
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)


    results_list = []
    trained_models_dict = {}
    
    fig_roc, ax_roc = plt.subplots(figsize=(8,6)); ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    fig_pr, ax_pr = plt.subplots(figsize=(8,6)); 
    chance_level_pr = y_test.mean() if not y_test.empty else 0.5
    ax_pr.plot([0, 1], [chance_level_pr, chance_level_pr], 'k--', label=f'Chance (AUC = {chance_level_pr:.2f})')

    for name, model_instance in models.items():
        st.subheader(f"Training and Evaluating: {name} for {disease_title}")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_instance)])
        current_model_results = {"Model": name}
        try:
            with st.spinner(f"Performing {cv_folds}-fold CV for {name}..."):
                cv_roc_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
                cv_pr_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
                current_model_results.update({
                    "CV ROC-AUC (Mean)": cv_roc_auc.mean(), "CV ROC-AUC (Std)": cv_roc_auc.std(),
                    "CV PR-AUC (Mean)": cv_pr_auc.mean(), "CV PR-AUC (Std)": cv_pr_auc.std()
                })
            pipeline.fit(X_train, y_train)
            trained_models_dict[name] = pipeline
            y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]
            current_model_results.update({
                "Test ROC-AUC": roc_auc_score(y_test, y_pred_proba_test),
                "Test PR-AUC": average_precision_score(y_test, y_pred_proba_test)
            })
            plot_roc_pr_curves(y_test, y_pred_proba_test, name, disease_title, ax_roc, ax_pr)

            with st.expander(f"Confusion Matrix & Metrics at Custom Threshold for {name} (Test Set) - {disease_title}"):
                thresh_key = f"threshold_slider_{name.replace(' ','_')}_{cv_key}_v3"
                classification_threshold = st.slider(
                    "Select Classification Threshold:", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.50,
                    step=0.01, 
                    key=thresh_key
                )
                
                y_pred_test_custom_threshold = (y_pred_proba_test >= classification_threshold).astype(int)
                cm = confusion_matrix(y_test, y_pred_test_custom_threshold)
                fig_cm_disp, ax_cm_disp = plt.subplots(); ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(ax=ax_cm_disp, cmap=plt.cm.Blues);
                ax_cm_disp.set_title(f'CM for {name} ({disease_title}, Thresh={classification_threshold:.2f})'); 
                st.pyplot(fig_cm_disp); plt.close(fig_cm_disp)
                
                tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]; tn = cm[0, 0]
                precision_manual = precision_score(y_test, y_pred_test_custom_threshold, zero_division=0)
                recall_manual = recall_score(y_test, y_pred_test_custom_threshold, zero_division=0)
                f1_manual = f1_score(y_test, y_pred_test_custom_threshold, zero_division=0)
                specificity_manual = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                st.markdown(f"""**Metrics at Threshold = {classification_threshold:.2f}:**
                             - Precision: {precision_manual:.3f}
                             - Recall (Sensitivity): {recall_manual:.3f}
                             - F1-Score: {f1_manual:.3f}
                             - Specificity: {specificity_manual:.3f}
                             - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}""")
                st.caption(f"""
                **How to Read the Confusion Matrix & Metrics:**
                -   **True Negative (TN):** Correctly predicted as 'No {disease_title}'.
                -   **False Positive (FP):** Incorrectly predicted as '{disease_title}' (Type I Error).
                -   **False Negative (FN):** Incorrectly predicted as 'No {disease_title}' (Type II Error - a "miss").
                -   **True Positive (TP):** Correctly predicted as '{disease_title}'.
                
                **Metric Explanations:**
                -   **Precision:** Out of all instances the model predicted as positive (TP + FP), what proportion was actually positive? 
                    Formula: TP / (TP + FP). 
                    *High precision means the model is trustworthy when it predicts positive.*
                -   **Recall (Sensitivity / True Positive Rate):** Out of all actual positive instances (TP + FN), what proportion did the model correctly identify? 
                    Formula: TP / (TP + FN). 
                    *High recall means the model is good at finding most of the positive instances.*
                -   **F1-Score:** The harmonic mean of Precision and Recall. It provides a balance between the two, especially useful when class distribution is uneven or when both false positives and false negatives are important. 
                    Formula: 2 * (Precision * Recall) / (Precision + Recall).
                -   **Specificity (True Negative Rate):** Out of all actual negative instances (TN + FP), what proportion did the model correctly identify? 
                    Formula: TN / (TN + FP).
                    *High specificity means the model is good at correctly identifying negative instances.*

                A good model generally aims for high values on the diagonal of the confusion matrix (TN, TP) and low values off-diagonal (FP, FN), leading to high Precision, Recall, F1-Score, and Specificity. The "best" metric to optimize for depends on the specific problem and the relative costs of different types of errors.
                """)
        except Exception as e:
            st.error(f"Error training/evaluating {name} for {disease_title}: {e}")
            current_model_results.update({k: np.nan for k in ["CV ROC-AUC (Mean)", "CV ROC-AUC (Std)", "CV PR-AUC (Mean)", "CV PR-AUC (Std)", "Test ROC-AUC", "Test PR-AUC"]})
        results_list.append(current_model_results)

    st.markdown(f"### Model Performance Metrics Summary for {disease_title}")
    if results_list:
        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df.style.format("{:.3f}", na_rep="Error", subset=pd.IndexSlice[:, results_df.columns.str.contains('AUC')]))

    ax_roc.set_title(f'ROC Curves for {disease_title} (Test Set)'); ax_roc.legend(loc='lower right'); st.pyplot(fig_roc); plt.close(fig_roc)
    ax_pr.set_title(f'Precision-Recall Curves for {disease_title} (Test Set)'); ax_pr.legend(loc='lower left'); st.pyplot(fig_pr); plt.close(fig_pr)

    # Logistic Regression Coefficients
    st.header(f"3. Logistic Regression Coefficients for {disease_title}")
    logreg_model = "Logistic Regression (L2)" 
    if logreg_model in trained_models_dict:
        logreg_pipeline = trained_models_dict[logreg_model]
        logreg_model_inspect = logreg_pipeline.named_steps['classifier']
        preprocessor_inspect = logreg_pipeline.named_steps['preprocessor']

        feature_names_transformed = []
        try:
            feature_names_transformed = preprocessor_inspect.get_feature_names_out()
        except AttributeError: 
            num_feat_names_coef = []
            if 'num' in preprocessor_inspect.named_transformers_ and hasattr(preprocessor_inspect.named_transformers_['num'], 'get_feature_names_out') and numerical_features:
                num_feat_names_coef = preprocessor_inspect.named_transformers_['num'].get_feature_names_out(numerical_features).tolist()
            elif 'num' in preprocessor_inspect.named_transformers_ and numerical_features: 
                 num_feat_names_coef = numerical_features 

            categorical_feature_name = []
            if 'cat' in preprocessor_inspect.named_transformers_ and categorical_features: 
                cat_transformer_obj = preprocessor_inspect.named_transformers_['cat']
                if hasattr(cat_transformer_obj, 'named_steps') and 'onehot' in cat_transformer_obj.named_steps:
                     onehot_encoder_obj = cat_transformer_obj.named_steps['onehot']
                     if categorical_features: 
                        categorical_feature_name = onehot_encoder_obj.get_feature_names_out(categorical_features).tolist()
                     else: 
                        categorical_feature_name = onehot_encoder_obj.get_feature_names_out().tolist()
                elif hasattr(cat_transformer_obj, 'get_feature_names_out'): 
                    categorical_feature_name = cat_transformer_obj.get_feature_names_out().tolist()
            feature_names_transformed = num_feat_names_coef + categorical_feature_name

        if hasattr(logreg_model_inspect, 'coef_') and len(feature_names_transformed) > 0:
            coefficients = logreg_model_inspect.coef_[0]
            if len(coefficients) == len(feature_names_transformed):
                coef_df = pd.DataFrame({'Feature': feature_names_transformed, 'Coefficient (Log-Odds)': coefficients})
                coef_df['Odds Ratio'] = np.exp(coef_df['Coefficient (Log-Odds)'])
                coef_df = coef_df.sort_values(by='Coefficient (Log-Odds)', key=abs, ascending=False)
                
                st.markdown(f"Coefficients from **{logreg_model}**. These represent the change in the log-odds of having {disease_title} for a one-unit change in the predictor, holding all other predictors in the model constant")
               
                
                st.dataframe(coef_df.style.format({'Coefficient (Log-Odds)': "{:.3f}", 'Odds Ratio': "{:.3f}"}))
                
                st.markdown("""
                **Interpreting Coefficients & Odds Ratios:**
                -   **Coefficient (Log-Odds):**
                    -   A positive coefficient means that as the feature value increases, the log-odds of having the disease increase (i.e., higher probability).
                    -   A negative coefficient means that as the feature value increases, the log-odds of having the disease decrease (i.e., lower probability).
                    -   A coefficient near zero suggests the feature has little independent linear association with the log-odds of the outcome in this model.
                -   **Odds Ratio (OR):** Calculated as `exp(coefficient)`.
                    -   OR > 1: For a one-unit increase in the feature, the odds of having the disease are multiplied by the OR. (e.g., OR=1.5 means 50% increased odds).
                    -   OR < 1: For a one-unit increase in the feature, the odds of having the disease are multiplied by the OR. (e.g., OR=0.7 means 30% decreased odds).
                    -   OR = 1: The feature has no effect on the odds of having the disease.
                -   **For Categorical Features (One-Hot Encoded):** The coefficient and OR are interpreted *relative to the reference category* (the one that was dropped during encoding). For example, if 'Income_Category_Low' has a positive coefficient, it means the Low Income category has higher log-odds of the disease compared to the reference income category.
                """)
            else:
                st.warning(f"Mismatch between number of coefficients and feature names.")
        else:
            st.warning(f"Feature names could not be extracted/are empty.")
    else:
        st.warning(f"Model not found")
    st.markdown("---")


    st.header(f"4. SHAP Value Visualization for {disease_title} Prediction")
    
    if not trained_models_dict: 
        st.warning(f"No models were successfully trained for {disease_title} to perform SHAP analysis.")
    else:
        try:
            shap_fitted = preprocessor.fit(X_train) 
            
            ohe_feature_names = []
            if categorical_features: 
                cat_transformer_in_pipe = shap_fitted.named_transformers_.get('cat')
                if cat_transformer_in_pipe:
                    onehot_step = cat_transformer_in_pipe.named_steps.get('onehot')
                    if onehot_step:
                        try: ohe_feature_names = onehot_step.get_feature_names_out(categorical_features).tolist()
                        except AttributeError: 
                            ohe_feature_names = onehot_step.get_feature_names(categorical_features).tolist()
            
            transformed_names = numerical_features + ohe_feature_names

            X_test_transformed = shap_fitted.transform(X_test)
            X_train_transformed = shap_fitted.transform(X_train)

            X_test_processed_shap = pd.DataFrame(X_test_transformed, columns=transformed_names, index=X_test.index)
            
            X_train_processed_shap_df = pd.DataFrame(columns=transformed_names) 
            if X_train_transformed.shape[0] > 0:
                num_background_samples = min(100, X_train_transformed.shape[0])
                background_data_np = shap.sample(X_train_transformed, num_background_samples, random_state=42)
                X_train_processed_shap_df = pd.DataFrame(background_data_np, columns=transformed_names)

            shap_model_key = f"shap_model_select_{cv_key}_v2"
            shap_model_name_selected = st.selectbox(f"Select model for SHAP ({disease_title}):", list(trained_models_dict.keys()), key=shap_model_key)
            
            selected_for_shap = trained_models_dict.get(shap_model_name_selected)
            if selected_for_shap:
                model_for_shap_analysis = selected_for_shap.named_steps['classifier']
                st.write(f"Calculating SHAP values for: **{shap_model_name_selected}** (Model type: `{type(model_for_shap_analysis).__name__}`)")

                with st.spinner(f"Generating SHAP plots for {shap_model_name_selected}..."):
                    explainer_obj = None; shap_values_calculated = None; shap_values_for_plotting = None
                    
                    if X_test_processed_shap.empty: st.warning("Test data for SHAP is empty.")
                    elif X_train_processed_shap_df.empty and isinstance(model_for_shap_analysis, (RandomForestClassifier, xgb.XGBClassifier, LogisticRegression)):
                        st.warning("Background data sample for SHAP is empty or too small.")
                    else:
                        if isinstance(model_for_shap_analysis, (RandomForestClassifier, xgb.XGBClassifier)):
                            explainer_obj = shap.TreeExplainer(model_for_shap_analysis, data=X_train_processed_shap_df, model_output="probability", feature_perturbation="interventional")
                            shap_values_calculated = explainer_obj.shap_values(X_test_processed_shap, check_additivity=False) 
                        elif isinstance(model_for_shap_analysis, LogisticRegression):
                            explainer_obj = shap.KernelExplainer(model_for_shap_analysis.predict_proba, X_train_processed_shap_df) 
                            shap_values_calculated = explainer_obj.shap_values(X_test_processed_shap, nsamples=min(500, X_test_processed_shap.shape[0]*2), check_additivity=False) 
                        
                        if shap_values_calculated is not None:
                            if isinstance(shap_values_calculated, list) and len(shap_values_calculated) == 2: 
                                shap_values_for_plotting = shap_values_calculated[1] 
                            elif isinstance(shap_values_calculated, np.ndarray) and shap_values_calculated.ndim == 2: 
                                shap_values_for_plotting = shap_values_calculated
                            elif isinstance(shap_values_calculated, np.ndarray) and shap_values_calculated.ndim ==3 and hasattr(explainer_obj, 'expected_value') and isinstance(explainer_obj.expected_value, (list, np.ndarray)) and len(explainer_obj.expected_value)==2 :
                                shap_values_for_plotting = shap_values_calculated[:,:,1] 
                    
                    if shap_values_for_plotting is not None and X_test_processed_shap.shape[0] > 0 and \
                       shap_values_for_plotting.shape[0] == X_test_processed_shap.shape[0] and \
                       shap_values_for_plotting.shape[1] == X_test_processed_shap.shape[1]:
                        
                        st.subheader(f"SHAP Summary Plot (Bar) for {disease_title} - {shap_model_name_selected}")
                        fig_shap_bar, ax_shap_bar = plt.subplots(); 
                        shap.summary_plot(shap_values_for_plotting, X_test_processed_shap, plot_type="bar", show=False, feature_names=X_test_processed_shap.columns); 
                        ax_shap_bar.set_title(f"SHAP Bar Plot: {shap_model_name_selected}"); st.pyplot(fig_shap_bar); plt.close(fig_shap_bar)
                        
                        st.subheader(f"SHAP Summary Plot (Beeswarm) for {disease_title} - {shap_model_name_selected}")
                        fig_shap_dot, ax_shap_dot = plt.subplots(); 
                        shap.summary_plot(shap_values_for_plotting, X_test_processed_shap, show=False, feature_names=X_test_processed_shap.columns); 
                        ax_shap_dot.set_title(f"SHAP Beeswarm Plot: {shap_model_name_selected}"); st.pyplot(fig_shap_dot); plt.close(fig_shap_dot)
                    else: st.warning(f"SHAP values for plotting are None or shape mismatch for {disease_title}. SHAP values shape: {getattr(shap_values_for_plotting, 'shape', 'N/A')}, X_test_processed shape: {X_test_processed_shap.shape}")
        except Exception as e_shap: 
            st.error(f"Error during SHAP analysis for {disease_title}: {e_shap}")
            st.exception(e_shap) 

    # Conditional SHAP Discussion 
    st.header(f"5. Discussion of L2 SHAP Plot Insights for {disease_title} Prediction") 
    with st.expander(f"Interpreting Feature Importance for {disease_title} Prediction", expanded=True):
        if disease_title == "Diabetes":
            st.markdown(f"""
            The SHAP plots help us understand which features are driving the model's predictions for **Diabetes**.
            """)
            st.markdown("""
            ### Features Increasing the Odds of Diabetes (Positive Association)

    * **Age:** This is by far the strongest positive predictor. As **age increases**, the odds of having diabetes **increase significantly** (Odds Ratio ~4.76). This is a well-established and expected biological risk factor.
    * **Body Mass Index (BMI):** A higher **BMI** is strongly associated with **increased odds of diabetes** (Odds Ratio ~1.74). This is another expected and significant risk factor.
    * **Gender:** Being **male** is associated with **higher odds of diabetes** compared to females (Odds Ratio ~1.20), holding other factors constant.
    * **Triglycerides:** Higher **triglyceride levels** are associated with **increased odds of diabetes** (Odds Ratio ~1.18), consistent with metabolic health indicators.
    * **Insulin:** Higher **insulin levels** are associated with **increased odds of diabetes** (Odds Ratio ~1.11). This is an expected finding, as elevated insulin often indicates insulin resistance, a precursor to Type 2 Diabetes.
    * **Systolic Blood Pressure:** Higher **systolic blood pressure** is associated with **increased odds of diabetes** (Odds Ratio ~1.10), indicating a slight but positive link.
    * **Minutes Sedentary Activity per Day:** More **sedentary time** is associated with **slightly increased odds of diabetes** (Odds Ratio ~1.06), highlighting the negative impact of inactivity.
    * **Diastolic Blood Pressure:** Higher **diastolic blood pressure** is associated with a **very slight increase in the odds of diabetes** (Odds Ratio ~1.03).
    
    ### Features Decreasing the Odds of Diabetes (Negative Association / Protective Factors)

    * **Has College Degree:** Having a **college degree** is associated with **lower odds of diabetes** (Odds Ratio ~0.82), suggesting a protective effect of higher education, likely due to factors like health literacy and access to resources.
    * **Direct HDL-Cholesterol:** Higher **HDL-C** (good cholesterol) is associated with **decreased odds of diabetes** (Odds Ratio ~0.84), which is an expected protective factor.
    * **Vigorous Recreational Activity:** Engaging in **vigorous recreational activity** is associated with **lower odds of diabetes** (Odds Ratio ~0.85), highlighting the protective effect of physical activity.
    * **Hemoglobin:** Higher **hemoglobin levels** are associated with **decreased odds of diabetes** (Odds Ratio ~0.89).
    * **Hours Sleep Weekdays:** More **sleep hours** on weekdays are associated with **slightly decreased odds of diabetes** (Odds Ratio ~0.95).
    
    ### Counter-Intuitive or Unexpected Findings and Potential Explanations

    In the adjusted logistic regression model, several features exhibited counter-intuitive associations with the odds of diabetes, warranting careful consideration. 
    For instance,  Total cholesterol (Odds Ratio ~0.84) was associated with lower odds of diabetes. 
    Similarly, increased Energy (kcal) intake (Odds Ratio ~0.85) also showed an unexpected association with decreased diabetes odds. 
    Furthermore, being a Current Smoker (Odds Ratio ~0.97) was linked to slightly lower odds of diabetes, which directly contradicts established medical understanding. 
    These surprising results might be attributed to multicollinearity, where highly correlated predictors obscure their individual effects, or confounding effects from unmeasured or complex interactions between variables in the multivariate model. For example, higher energy intake might be correlated with higher physical activity, which is protective, leading the model to misattribute the protective effect.
    

            """)
            st.markdown("""
            **General SHAP Interpretation Reminder:**
            -   Bar Plot: Overall feature importance.
            -   Beeswarm Plot: Direction and distribution of feature effects.
            -   These are associations learned by the model from *this specific dataset*. They do not automatically imply causation. Cross-sectional data has limitations (like reverse causality).
            """)

        elif disease_title == "Hypertension": 
            st.markdown(f"""
            The SHAP plots help us understand which features are driving the model's predictions for **Hypertension**.
            """)
            st.markdown("""
            ### Features Increasing the Odds of Hypertension (Positive Association)

            * **Age:** This is the strongest positive predictor. As **age increases**, the odds of having hypertension **increase significantly** (Odds Ratio ~3.54), a well-established biological risk factor.
            * **Body Mass Index (BMI):** A higher **BMI** is strongly associated with **increased odds of hypertension** (Odds Ratio ~1.70), consistent with expected links between obesity and blood pressure.
            * **Insulin:** Higher **insulin levels** are associated with **increased odds of hypertension** (Odds Ratio ~1.12), pointing to connections between metabolic health and blood pressure regulation.
            * **Current Smoker:** Being a **current smoker** is associated with **increased odds of hypertension** (Odds Ratio ~1.12).
            * **Minutes Sedentary Activity per Day:** More **sedentary time** is associated with **increased odds of hypertension** (Odds Ratio ~1.11), highlighting the negative impact of inactivity.
            * **Gender:** Being **male** is associated with **slightly higher odds of hypertension** compared to females (Odds Ratio ~1.07), holding other factors constant.
            * **Triglycerides:** Higher **triglyceride levels** are associated with **increased odds of hypertension** (Odds Ratio ~1.07).
            * **Alcohol (gm):** Higher **alcohol consumption** is associated with **increased odds of hypertension** (Odds Ratio ~1.05).
            * **Energy (kcal):** Higher **caloric intake** shows a slight **increase in the odds of hypertension** (Odds Ratio ~1.04).

            ### Features Decreasing the Odds of Hypertension (Negative Association / Protective Factors)

            * **Hemoglobin:** Higher **hemoglobin levels** are associated with **lower odds of hypertension** (Odds Ratio ~0.86).
            * **Has College Degree:** Having a **college degree** is associated with **lower odds of hypertension** (Odds Ratio ~0.89), suggesting a protective effect of higher education.
            * **Hours Sleep Weekdays:** More **sleep hours** on weekdays are associated with **lower odds of hypertension** (Odds Ratio ~0.93).
            * **Direct HDL-Cholesterol:** Higher **HDL-C** (good cholesterol) is associated with **lower odds of hypertension** (Odds Ratio ~0.95), an expected protective factor.
            * **PIR_Ratio:** A higher **PIR (Poverty Income Ratio)** is associated with **lower odds of hypertension** (Odds Ratio ~0.96).

            ### Counter-Intuitive or Unexpected Findings and Potential Explanations

            In the adjusted logistic regression model for hypertension, certain features exhibited associations that were unexpected based on common medical understanding, necessitating careful consideration:

            * **Sodium (mg):** This model suggests that higher **sodium intake** is associated with **slightly lower odds of hypertension** (Odds Ratio ~0.98).
                * **Potential Explanation:** This is highly counter-intuitive given the established link between high sodium and hypertension. This result almost certainly indicates **reverse causality** (individuals already diagnosed with hypertension might have reduced their sodium intake, making the current cross-sectional data show a lower correlation or even inverse link), **multicollinearity** with other dietary components, or **confounding effects** not fully captured by the model.
            * **Total cholesterol (mg/dL):** Higher **total cholesterol** is associated with **slightly lower odds of hypertension** (Odds Ratio ~0.98).
                * **Potential Explanation:** Similar to sodium, this unexpected inverse relationship could be due to **multicollinearity** within the lipid profile or **confounding**. It might also reflect complex interactions in an adjusted model where other features account for related risk.
            * **Vigorous Recreational Activity (Yes/No):** Engaging in **vigorous recreational activity** is associated with **slightly lower odds of hypertension** (Odds Ratio ~0.98), but the effect is very weak.
                * **Potential Explanation:** While exercise is generally protective, the very small effect size and borderline protective OR might indicate that its impact is largely captured by other highly correlated lifestyle factors, or that the binary representation (Yes/No) doesn't fully capture the dose-response relationship, or that the model is adjusting for other factors that mediate this relationship.

            """)
            st.markdown("""
            **General SHAP Interpretation Reminder:**
            - Bar Plot: Overall feature importance.
            - Beeswarm Plot: Direction and distribution of feature effects.
            - These are associations learned by the model from *this specific dataset*. They do not automatically imply causation. Cross-sectional data has limitations (like reverse causality).
            """)