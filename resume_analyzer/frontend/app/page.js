"use client";

import { useState } from 'react';
import Navbar from '../components/Navbar';
import FileUpload from '../components/FileUpload';
import ResultsCard from '../components/ResultsCard';
import ParsedView from '../components/ParsedView';
import { uploadResume, scoreResume, parseResume } from '../lib/api';
import { Loader2, ArrowRight } from 'lucide-react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [parsedResults, setParsedResults] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!file || !jobDescription) {
      setError("Please upload a resume and provide a job description.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      // Execute both requests in parallel
      const [scoreData, parseData] = await Promise.all([
        scoreResume(file, jobDescription),
        parseResume(file)
      ]);

      setResults(scoreData);
      setParsedResults({ ...parseData.parsed_data, filename: parseData.filename });
    } catch (err) {
      console.error(err);
      setError("Failed to analyze resume. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white selection:bg-green-500/30">
      <Navbar />
      
      <main className="max-w-7xl mx-auto px-6 py-12 space-y-12">
        {/* Hero Section */}
        <div className="text-center space-y-4 max-w-2xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent">
            Optimize Your <span className="text-green-500">Resume</span>
          </h1>
          <p className="text-zinc-400 text-lg">
            AI-powered analysis to beat the ATS and land your dream job.
            Get instant role predictions and scoring.
          </p>
        </div>

        {/* Input Section */}
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
          <FileUpload onFileSelect={(f) => { setFile(f); setResults(null); setError(null); }} />

          <div className="max-w-xl mx-auto">
            <label className="block text-sm font-medium text-zinc-400 mb-2">
              Job Description
            </label>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job description here..."
              className="w-full h-32 bg-zinc-900 border border-zinc-800 rounded-xl p-4 text-white placeholder-zinc-600 focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 transition-all resize-none"
            />
          </div>

          <div className="flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={loading || !file || !jobDescription}
              className={`
                group flex items-center gap-2 px-8 py-4 rounded-full font-bold text-lg transition-all
                ${loading || !file || !jobDescription
                  ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                  : "bg-green-500 text-black hover:bg-green-400 hover:scale-105 shadow-[0_0_20px_-5px_rgba(34,197,94,0.4)]"
                }
              `}
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Resume
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </div>
          
          {error && (
            <div className="text-red-500 text-center bg-red-500/10 p-4 rounded-xl max-w-xl mx-auto border border-red-500/20">
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="flex items-center justify-center gap-2 text-green-500 mb-8">
               <div className="h-px w-12 bg-green-500/50" />
               <span className="text-sm font-semibold uppercase tracking-wider">Analysis Results</span>
               <div className="h-px w-12 bg-green-500/50" />
            </div>
            
            
            <ResultsCard 
              atsResult={results.ats_score} 
              rolePrediction={results.role_prediction} 
            />
            
            <ParsedView parsedData={parsedResults} />
          </div>
        )}
      </main>
    </div>
  );
}
