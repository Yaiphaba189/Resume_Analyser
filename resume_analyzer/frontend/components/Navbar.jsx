import Link from 'next/link';
import { Terminal } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="w-full border-b border-zinc-800 bg-black/50 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="p-2 rounded-lg bg-zinc-900 group-hover:bg-green-500/10 transition-colors">
            <Terminal className="w-6 h-6 text-green-500 group-hover:text-green-400" />
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent group-hover:from-green-400 group-hover:to-white transition-all">
            ResumeAnalyzer
          </span>
        </Link>
        
        <div className="flex items-center gap-6 text-sm font-medium text-zinc-400">
          <Link href="/" className="hover:text-green-400 transition-colors">Analyzer</Link>
        </div>
      </div>
    </nav>
  );
}
