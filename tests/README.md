# Heart Disease MLOps Pipeline

End-to-end MLOps pipeline for predicting heart disease risk based on patient clinical parameters.

## Repository Structure
- `app.py`: FastAPI inference service.
- `Dockerfile`: Production container build specification using `python:3.8-slim`.
- `requirements.txt`: Project dependencies (`fastapi`, `uvicorn`, `scikit-learn`, `pytest`, etc.).
- `model.pkl` & `scaler.pkl`: Trained scikit-learn model and scaler artifacts.
- `k8s/`: Kubernetes manifests (`deployment.yaml` and `service.yaml`).
- `tests/test_pipeline.py`: Automated unit tests for pipeline components.
- `.github/workflows/main.yml`: CI/CD automation workflow.

## Verification & Execution Steps

### 1. Run Unit Tests
```bash
pytest tests/