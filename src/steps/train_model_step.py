import os.path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenml.steps import step


# Define the model at the module level, so it's pickleable
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


@step
def train_model() -> str:
    if os.path.exists('../../model/preprocessor.pkl') and os.path.exists('../../model/sphinx_value_vision.pt'):
        return ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read the dataset
    df = pd.read_csv(r"../../data/dataset.csv")
    features = [
        'area', 'rooms', 'bathrooms', 'style', 'floor', 'year_built', 'seller_type',
        'view', 'payment_method', 'location', 'apartment_age'
    ]
    X = df[features]
    y = df['price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numerical columns
    categorical_cols = ['location', 'style', 'seller_type', 'view', 'payment_method']
    numerical_cols = [col for col in features if col not in categorical_cols]

    # Pipelines for preprocessing
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, '../../model/preprocessor.pkl')

    # Convert to torch tensors
    if hasattr(X_train_processed, 'toarray'):
        X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32).to(device)
    else:
        X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    # DataLoader setup
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model and move to device
    model = RegressionNN(X_train_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping parameters
    best_loss = np.inf
    patience = 10
    trigger_times = 0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation after each epoch
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).detach().cpu().numpy()
            y_true = y_test_tensor.detach().cpu().numpy()
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

        print(f"Epoch {epoch + 1}/{epochs} | R²: {r2:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if mse < best_loss:
            best_loss = mse
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

    # Final evaluation
    final_preds = model(X_test_tensor).detach().cpu().numpy()
    final_true = y_test_tensor.detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(final_true, final_preds))
    mae = mean_absolute_error(final_true, final_preds)
    r2_final = r2_score(final_true, final_preds)
    print("\nFinal Neural Network Model Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2_final:.4f}")

    # Convert the model to TorchScript
    scripted_model = torch.jit.script(model)
    model_path = "../../model/sphinx_value_vision.pt"  # Optional: change extension to .pt
    scripted_model.save(model_path)

    return model_path


# if __name__ == '__main__':
#     train_model()


# Final Neural Network Model Evaluation:
# RMSE: 459723.10
# MAE: 300520.94
# R²: 0.8321
