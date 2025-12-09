# Resume Analyzer Frontend

A modern, AI-powered Resume Analyzer frontend built with Next.js 16, Tailwind CSS, and Framer Motion. This application interfaces with a backend API to score resumes against job descriptions and predict suitable job roles.

## Features

- **Resume Upload**: Drag-and-drop support for PDF, DOCX, and image files.
- **ATS Scoring**: Visual gauge chart showing the Antit-Tracking System (ATS) compatibility score.
- **Role Prediction**: AI-driven prediction of the most suitable job role based on resume content.
- **Missing Keywords**: Highlights key terms missing from the resume compared to the job description.
- **Responsive Design**: Fully responsive UI with a premium dark mode aesthetic.

## Tech Stack

- **Framework**: [Next.js 16](https://nextjs.org/) (App Directory)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Animations**: [Framer Motion](https://www.framer.com/motion/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **HTTP Client**: [Axios](https://axios-http.com/)

## Getting Started

### Prerequisites

- Node.js 18+ installed.
- Backend server running on `http://localhost:8000`.

### Installation

1. Navigate to the frontend directory:

   ```bash
   cd resume_analyzer/frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

- `app/`: Next.js App Router pages and layouts.
- `components/`: Reusable UI components (ResultsCard, FileUpload, Navbar, etc.).
- `lib/`: Utility functions and API client configuration (`api.js`).

## Usage

1. **Upload Resume**: Click or drag your resume file onto the upload area.
2. **Add Job Description**: Paste the job description you are applying for.
3. **Analyze**: Click the "Analyze Resume" button.
4. **View Results**: Check your ATS score, predicted role, and missing keywords in the results dashboard.
