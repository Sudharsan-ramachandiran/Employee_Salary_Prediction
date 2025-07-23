import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess(df):
    df = df.dropna()
    df = df[df['salary_in_usd'] < 300000]
    df['log_salary'] = np.log1p(df['salary_in_usd'])

    features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
    X = df[features]
    y = df['log_salary']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(features))
    X_encoded_df.index = df.index

    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder
