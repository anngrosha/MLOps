import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_and_evaluate():
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    X_train = train_data.drop('loan_status', axis=1)
    y_train = train_data['loan_status']

    X_test = test_data.drop('loan_status', axis=1)
    y_test = test_data['loan_status']

    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")

        pickle.dump(model, open('models/model.pkl', 'wb'))
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    train_and_evaluate()
