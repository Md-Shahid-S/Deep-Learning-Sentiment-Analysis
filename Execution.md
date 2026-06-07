# 🛠️ Project Execution Guide

Follow these steps to set up, train, and run the entire Sentiment Analysis project.

## 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 4GB+ RAM (for training Deep Learning models)

## 2. Local Setup (Without Docker)

### A. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'])"
   ```

### B. Train the Models
You must train the models before running the API to generate the necessary artifacts.
1. Train Baseline Models (SVM & Logistic Regression):
   ```bash
   python -m training.train_baselines
   ```
2. Train LSTM Model:
   ```bash
   python -m training.train_lstm
   ```
*Artifacts will be saved in `backend/artifacts/`.*

### C. Run the Backend
```bash
uvicorn app.main:app --reload
```

### D. Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 3. Docker Execution (Recommended)

To run the entire stack in one go using Docker:

1. **Train models locally first** (optional but recommended if you don't want to train inside a container) OR ensure the `artifacts` folder is populated.
2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
3. Access the applications:
   - **Frontend:** [http://localhost:8501](http://localhost:8501)
   - **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 4. Troubleshooting
- **Model not loaded:** Ensure that the `artifacts/` folder contains `.h5` and `.joblib` files.
- **Connection Error:** If the frontend cannot reach the backend, check the `API_URL` in `.env` or `docker-compose.yaml`.
- **NLTK errors:** Run the NLTK download command mentioned in the setup section.
