import logging
import torch
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the request schema without 'price' (it's the target, not a feature)
class PredictionRequest(BaseModel):
    location: str
    area: float
    rooms: int
    bathrooms: int
    style: str
    floor: int
    year_built: int
    seller_type: str
    view: str
    payment_method: str


# Load the preprocessor
try:
    preprocessor = joblib.load('model/preprocessor.pkl')
    logger.info("Preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the preprocessor: {e}")
    preprocessor = None

# Load the TorchScript model
try:
    model = torch.jit.load('model/sphinx_value_vision.pt', map_location=torch.device('cpu'))
    model.eval()  # Good practice, though not strictly necessary for TorchScript
    logger.info("TorchScript model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the TorchScript model: {e}")
    model = None


# Exception handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(exc: RequestValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def general_exception_handler(exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )


@app.post("/predict")
async def predict(data: PredictionRequest):
    # Ensure the preprocessor and model are loaded
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded.")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Compute apartment_age (assuming 2025 as the current year, consistent with training)
        current_year = 2025
        apartment_age = current_year - data.year_built

        # Create a DataFrame with features in the order expected by the preprocessor
        input_data = {
            'area': data.area,
            'rooms': data.rooms,
            'bathrooms': data.bathrooms,
            'style': data.style,
            'floor': data.floor,
            'year_built': data.year_built,
            'seller_type': data.seller_type,
            'view': data.view,
            'payment_method': data.payment_method,
            'location': data.location,
            'apartment_age': apartment_age
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess the data using the loaded preprocessor
        processed_data = preprocessor.transform(input_df)

        # Convert to torch tensor (handle sparse output from OneHotEncoder if applicable)
        if hasattr(processed_data, 'toarray'):
            input_tensor = torch.tensor(processed_data.toarray(), dtype=torch.float32)
        else:
            input_tensor = torch.tensor(processed_data, dtype=torch.float32)

        # Run the model prediction
        with torch.no_grad():
            prediction_tensor = model(input_tensor)

        # Extract the predicted value (assuming a single output)
        prediction_value = prediction_tensor.item()

        return {
            "prediction": prediction_value,
        }
    except Exception as err:
        logger.error(f"Error in prediction endpoint: {err}")
        raise HTTPException(status_code=500, detail="Error during prediction.")
