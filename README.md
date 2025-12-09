# Resume Analyzer Powered by AI

A comprehensive Application Tracking System (ATS) and Resume Analyzer solution that leverages Advanced NLP, Machine Learning, and Modern Web Technologies to help candidates optimize their resumes and recruiters find the best talent.

## 📂 Project Structure

This repository is a monorepo containing both the backend and frontend services:

- **[Backend](resume_analyzer/backend/README.md)**: A FastAPI-based service handling resume parsing, OCR text extraction, ATS scoring, and ML-based job role prediction.
- **[Frontend](resume_analyzer/frontend/README.md)**: A Next.js 16 web application providing a modern, responsive UI for users to upload resumes and view insights.

## 🚀 Quick Start

To run the full application locally, you will need to start both the backend and frontend servers.

### 1. Start the Backend

```bash
cd resume_analyzer/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`.

### 2. Start the Frontend

```bash
cd resume_analyzer/frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## ✨ Key Features

- **Resume Parsing**: Extract details like Name, Email, Skills, and Sections.
- **ATS Scoring**: Get a compatibility score (0-100%) against a specific Job Description.
- **Job Role Prediction**: AI model predicts the most suitable job role for the resume.
- **Missing Keywords**: Identify critical skills or keywords missing from the resume.
- **OCR Support**: Handles image-based resumes and non-selectable PDFs.

## 🛠️ Technologies

- **Backend**: Python, FastAPI, TensorFlow, Spacy, OpenCV, Pypdf.
- **Frontend**: Next.js 16, TailwindCSS, Framer Motion, React.
