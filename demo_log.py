from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# FastAPI app
app = FastAPI()

# Load Iris model
iris = load_iris()
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(iris.data, iris.target)

# Iris species names
species_names = ['setosa', 'versicolor', 'virginica']

def iris_model(features: dict):
    """
    Predict Iris species from features
    Expected keys: sepal_length, sepal_width, petal_length, petal_width
    OR: feature1, feature2, feature3, feature4
    """
    time.sleep(0.1)  # Simulate compute
    
    # Extract features in correct order
    if 'sepal_length' in features:
        feature_array = np.array([[
            features['sepal_length'],
            features['sepal_width'],
            features['petal_length'],
            features['petal_width']
        ]])
    else:
        feature_array = np.array([[
            features.get('feature1', 0),
            features.get('feature2', 0),
            features.get('feature3', 0),
            features.get('feature4', 0)
        ]])
    
    # Predict
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = float(probabilities.max())
    
    return {
        "prediction": int(prediction),
        "species": species_names[prediction],
        "confidence": round(confidence, 4),
        "probabilities": {
            "setosa": round(float(probabilities[0]), 4),
            "versicolor": round(float(probabilities[1]), 4),
            "virginica": round(float(probabilities[2]), 4)
        }
    }

# Input schema
class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# App state
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    time.sleep(2)  # Simulate model loading
    app_state["is_ready"] = True
    logger.info(json.dumps({"event": "startup_complete", "status": "ready"}))

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(input: Input, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = input.dict()
            result = iris_model(input_data)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            return result

        except Exception as e:
            logger.error(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
