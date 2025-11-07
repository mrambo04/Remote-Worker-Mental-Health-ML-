import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

# Load model and encoders
model=joblib.load('random_forest.pkl')
burn_map=joblib.load('burn_map.pkl')
salary_enc=joblib.load('salary_order.pkl')
health_classes=joblib.load('health_pkl')
nom=joblib.load('nominal.pkl')
label_enc=joblib.load('lable.pkl')

# Load saved dummy columns list (this is a list, not an encoder)
nominal_enc = joblib.load("C:/Users/nisha/dummy_columns.pkl")

# Load sample data for dropdown options
inp = pd.read_csv("C:/Users/nisha/Downloads/mental.csv")

menu = st.sidebar.radio("Navigation", ["📊 Dataset Overview", "🧠 Mental Health Prediction"])

if menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview & Insights")
    st.write("Here’s a summary of the dataset used for training the burnout prediction model.")
    
    # Shape of dataset
    st.subheader("Dataset Shape")
    st.write(f"Rows: {inp.shape[0]}, Columns: {inp.shape[1]}")

st.subheader("Sample Data")
st.dataframe(inp.head())

    # Column descriptions (manually prepared)
st.subheader("Column Descriptions")
col_desc = {
        "Age": "Age of the employee in years",
        "Gender": "Male / Female / Other",
        "Region": "Geographic work location",
        "Industry": "Type of industry the employee works in",
        "Job_Role": "Specific job role or designation",
        "Work_Arrangement": "Work setting (Remote, Hybrid, Onsite)",
        "Hours_Per_Week": "Total work hours per week",
        "Burnout_Level": "Employee burnout status (target variable)",
        "Work_Life_Balance": "Work-life balance satisfaction score",
        "Salary_Range": "Salary range category",
        "Physical_Health_Issues": "List of health issues faced by employee"
    }

col_df = pd.DataFrame(list(col_desc.items()), columns=["Column", "Description"])
st.table(col_df)

st.title("Mental Health Prediction")

# User inputs
age = st.number_input("Age", 18, 80, 30)
gender = st.selectbox("Gender", inp.Gender.unique())
region = st.selectbox("Region", inp.Region.unique())
industry = st.selectbox("Industry", inp.Industry.unique())
job_role = st.selectbox("Job Role", inp.Job_Role.unique())
work_arrangement = st.selectbox("Work Arrangement", inp.Work_Arrangement.unique())
hours_per_week = st.number_input("Hours Per Week", 0, 100, 40)
burnout_level = st.selectbox("Burnout Level", inp.Burnout_Level.unique())
work_life_balance = st.number_input("Work Life Balance Score", 1, 5, 3)
physical_health_issues = st.multiselect(
    "Physical Health Issues",
    ['Back Pain', 'Eye Strain', 'Neck Pain', 'Shoulder Pain', 'Wrist Pain', 'None']
)
social_isolation_score = st.number_input("Social Isolation Score", 1, 5, 3)
salary_range = st.selectbox("Salary Range", inp.Salary_Range.unique())

if st.button("Predict"):
    # Build input DataFrame for one row
    input_df = pd.DataFrame([[
        age, gender, region, industry, job_role, work_arrangement, hours_per_week,
        burnout_level, work_life_balance, "; ".join(physical_health_issues),
        social_isolation_score, salary_range
    ]], columns=inp.columns)

    # Encode Burnout_Level
    if hasattr(burn_map, 'transform'):
        input_df['Burnout_Level'] = burn_map.transform(input_df[['Burnout_Level']])
    else:
        input_df['Burnout_Level'] = input_df['Burnout_Level'].map(burn_map)

    # Encode Salary_Range
    if hasattr(salary_enc, 'transform'):
        input_df['Salary_Range'] = salary_enc.transform(input_df[['Salary_Range']])
    else:
        input_df['Salary_Range'] = input_df['Salary_Range'].map(salary_enc)

    # Encode Physical_Health_Issues
    input_df['Physical_Health_Issues'] = input_df['Physical_Health_Issues'].fillna('None')
    input_df['Physical_Health_Issues'] = input_df['Physical_Health_Issues'].apply(
        lambda x: x.split('; ') if isinstance(x, str) and x != 'None' else []
    )
    mlb = MultiLabelBinarizer(classes=list(health_classes))
    mlb.fit([])  # Initialize MultiLabelBinarizer
    health_encoded = mlb.transform(input_df['Physical_Health_Issues'])
    health_df = pd.DataFrame(health_encoded, columns=mlb.classes_, index=input_df.index)

    # Merge health encoded columns and drop original
    input_df = pd.concat([input_df.drop('Physical_Health_Issues', axis=1), health_df], axis=1)

    # Nominal columns to encode
    nominal_cols = ['Gender', 'Region', 'Industry', 'Job_Role', 'Work_Arrangement']

    # Get dummies for nominal columns (drop_first=True or False depending on training)
    input_nominal_dummies = pd.get_dummies(input_df[nominal_cols], drop_first=True)

    # Add missing dummy columns (those from training but missing here)
    for col in nominal_enc:
        if col not in input_nominal_dummies.columns:
            input_nominal_dummies[col] = 0

    # Reorder columns to match training dummy columns order
    input_nominal_dummies = input_nominal_dummies[nominal_enc]

    # Drop original nominal columns and add dummy columns
    input_df = input_df.drop(columns=nominal_cols)
    input_df = pd.concat([input_df.reset_index(drop=True), input_nominal_dummies.reset_index(drop=True)], axis=1)

    # Ensure input_df has all model features in exact order
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Predict
    prediction_encoded = model.predict(input_df)

    # Decode prediction label
    if hasattr(label_enc, 'inverse_transform'):
        prediction = label_enc.inverse_transform(prediction_encoded)
    else:
        prediction = prediction_encoded

    st.subheader(f"Predicted Mental Health Status: {prediction[0]}")

