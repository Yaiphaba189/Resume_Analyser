import { motion } from 'framer-motion';
import { User, Mail, Phone, Code, BookOpen } from 'lucide-react';

export default function ParsedView({ parsedData }) {
  if (!parsedData) return null;

  const { name, email, phone, skills, education, experience } = parsedData;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-4xl mx-auto mt-8 bg-zinc-900/30 border border-zinc-800 rounded-3xl p-8"
    >
      <h3 className="text-zinc-400 font-medium mb-6 flex items-center gap-2">
        <User className="w-5 h-5" /> Candidate Profile
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h4 className="text-xl font-bold text-white mb-1">{name || parsedData.filename || "Unknown Candidate"}</h4>
          <div className="flex flex-col gap-2 text-zinc-400 text-sm mt-1">
             <span className="italic opacity-60">Candidate Report</span>
          </div>
        </div>

        <div>
          <h5 className="text-zinc-500 text-sm mb-3 flex items-center gap-2">
            <Phone className="w-4 h-4" /> Contact Information
          </h5>
          <div className="flex flex-col gap-3 bg-zinc-800/50 p-4 rounded-xl border border-zinc-700/50">
             {phone ? (
                <div className="flex items-center gap-3 text-zinc-300">
                    <div className="p-2 bg-green-500/10 rounded-lg text-green-500">
                        <Phone className="w-4 h-4" />
                    </div>
                    <span className="font-medium">{phone}</span>
                </div>
             ) : (
                <div className="text-zinc-500 text-sm italic">No phone number detected</div>
             )}
             
             {email && (
                <div className="flex items-center gap-3 text-zinc-300">
                    <div className="p-2 bg-blue-500/10 rounded-lg text-blue-500">
                        <Mail className="w-4 h-4" />
                    </div>
                    <span className="font-medium">{email}</span>
                </div>
             )}
          </div>

          <h5 className="text-zinc-500 text-sm mb-3 mt-6 flex items-center gap-2">
            <Code className="w-4 h-4" /> Skills
          </h5>
          <div className="flex flex-wrap gap-2">
            {skills && skills.length > 0 ? (
              skills.map((skill, idx) => (
                <span key={idx} className="px-3 py-1 rounded-full bg-zinc-800 text-zinc-300 text-sm border border-zinc-700">
                  {skill}
                </span>
              ))
            ) : (
              <span className="text-zinc-600 italic">No skills specific detected</span>
            )}
          </div>
        </div>
      </div>
      
      {/* Additional Sections */}
      <div className="mt-8 grid grid-cols-1 gap-6">
        {parsedData.summary && (
          <SectionView title="Professional Summary" icon={<User className="w-4 h-4" />}>
            {parsedData.summary}
          </SectionView>
        )}

        {parsedData.experience && (
          <SectionView title="Work Experience" icon={<BookOpen className="w-4 h-4" />}>
            {parsedData.experience}
          </SectionView>
        )}

        {parsedData.education && (
          <SectionView title="Education" icon={<BookOpen className="w-4 h-4" />}>
            {parsedData.education}
          </SectionView>
        )}
      </div>
    </motion.div>
  );
}

function SectionView({ title, icon, children }) {
  // Split the content into lines and filter empty ones
  const lines = typeof children === 'string' 
    ? children.split('\n').filter(line => line.trim().length > 0)
    : [];

  return (
    <div className="bg-zinc-800/50 rounded-xl p-6 border border-zinc-700/50">
      <h5 className="text-zinc-400 text-sm font-medium mb-4 flex items-center gap-2">
        {icon} {title}
      </h5>
      <ul className="space-y-2">
        {lines.map((line, idx) => {
          // HEURISTIC: Check if line starts with a common bullet char
          const cleanLine = line.trim();
          const isBullet = cleanLine.startsWith('•') || cleanLine.startsWith('-') || cleanLine.startsWith('*');
          
          return (
            <li key={idx} className={`text-zinc-300 text-sm leading-relaxed flex gap-2 ${isBullet ? 'pl-2' : ''}`}>
              {!isBullet && <span className="text-zinc-600 block mt-1.5 w-1.5 h-1.5 rounded-full bg-zinc-600 flex-shrink-0" />}
              <span>{cleanLine}</span>
            </li>
          )
        })}
      </ul>
    </div>
  );
}
