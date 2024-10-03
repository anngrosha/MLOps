import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler

def data_prep(data):
    data.columns = data.columns.str.strip()
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    data.drop(columns=['loan_id'], inplace=True)
    data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
    data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
    data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

    columns_to_normalize = [
        'no_of_dependents', 'income_annum', 'loan_amount', 
        'loan_term', 'cibil_score', 'residential_assets_value', 
        'commercial_assets_value', 'luxury_assets_value', 
        'bank_asset_value'
    ]

    scaler = StandardScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    pickle.dump(scaler, open('MLOps/models/scaler.pkl', 'wb'))

    return data

data = pd.read_csv('MLOps/code/datasets/loan_approval_dataset.csv')

data = data_prep(data)

data.to_csv('MLOps/data/normalized_data.csv', index=False)

X = data.drop('loan_status', axis = 1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('MLOps/models/model.pkl', 'wb'))
