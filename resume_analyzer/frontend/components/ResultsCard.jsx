import { motion } from 'framer-motion';
import { Award, Briefcase, Zap, CheckCircle2, AlertCircle } from 'lucide-react';

export default function ResultsCard({ atsResult, rolePrediction }) {
  const atsScore = atsResult?.final_ats_score || 0;
  const missingKeywords = atsResult?.missing_keywords || [];
  
  const scoreColor = atsScore >= 80 ? 'text-green-500' : atsScore >= 50 ? 'text-yellow-500' : 'text-red-500';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6"
    >
      {/* ATS Score Card */}
      <div className="bg-zinc-900/50 backdrop-blur border border-zinc-800 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden group">
        <div className={`absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity duration-500 bg-gradient-to-br from-green-500 to-transparent`} />
        
        <h3 className="text-zinc-400 font-medium mb-4 flex items-center gap-2">
          <Award className="w-5 h-5" /> ATS Score
        </h3>
        
        <div className="relative">
          <svg className="w-48 h-48 transform -rotate-90">
            <circle
              className="text-zinc-800"
              strokeWidth="12"
              stroke="currentColor"
              fill="transparent"
              r="70"
              cx="96"
              cy="96"
            />
            <circle
              className={scoreColor}
              strokeWidth="12"
              strokeDasharray={440}
              strokeDashoffset={440 - (440 * atsScore) / 100}
              strokeLinecap="round"
              stroke="currentColor"
              fill="transparent"
              r="70"
              cx="96"
              cy="96"
            />
          </svg>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
            <span className={`text-5xl font-bold ${scoreColor}`}>
              {Math.round(atsScore)}%
            </span>
          </div>
        </div>
        
        <p className="mt-4 text-center text-zinc-500 text-sm">
          {atsScore >= 80 ? "Excellent Match!" : atsScore >= 50 ? "Good Potential" : "Needs Improvement"}
        </p>
      </div>

      {/* Role Prediction Card */}
      <div className="bg-zinc-900/50 backdrop-blur border border-zinc-800 rounded-3xl p-8 relative overflow-hidden">
        <h3 className="text-zinc-400 font-medium mb-6 flex items-center gap-2">
          <Briefcase className="w-5 h-5" /> Role Prediction
        </h3>

        {rolePrediction && (
          <div className="space-y-6">
            <div>
              <p className="text-sm text-zinc-500 mb-1">Top Prediction</p>
              <div className="text-2xl font-bold text-white flex items-center gap-3">
                {rolePrediction.top_role}
                <span className="text-xs font-normal px-2 py-1 rounded-full bg-green-500/10 text-green-500 border border-green-500/20">
                  {Math.round(rolePrediction.probabilities[0] * 100)}% Confidence
                </span>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-sm text-zinc-500">Alternative Matches</p>
              {rolePrediction.top_3_roles.slice(1).map((role, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-xl bg-black/40 border border-zinc-800">
                  <span className="text-zinc-300">{role}</span>
                  <span className="text-zinc-500 text-sm">{Math.round(rolePrediction.probabilities[idx + 1] * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Missing Keywords / Insights */}
      <div className="md:col-span-2 bg-zinc-900/30 border border-zinc-800 rounded-3xl p-6">
        <h3 className="text-zinc-400 font-medium mb-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" /> Missing Keywords
        </h3>
        
        {missingKeywords.length > 0 ? (
           <div className="flex flex-wrap gap-2">
              {missingKeywords.map((kw, idx) => (
                 <span key={idx} className="px-3 py-1 rounded-full bg-red-500/10 text-red-500 text-sm border border-red-500/20">
                    {kw}
                 </span>
              ))}
           </div>
        ) : (
           <div className="flex items-center gap-3 text-zinc-300">
             <CheckCircle2 className="w-5 h-5 text-green-500" />
             <p className="text-sm">Great job! No major keywords extracted from the JD seem to be missing.</p>
           </div>
        )}
      </div>
    </motion.div>
  );
}
