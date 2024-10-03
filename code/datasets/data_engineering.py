import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def data_load():
    data = pd.read_csv('data/raw/loan_approval_dataset.csv')

    data.columns = data.columns.str.strip()
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    data.dropna(inplace=True)

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

    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)

if __name__ == "__main__":
    data_load()
