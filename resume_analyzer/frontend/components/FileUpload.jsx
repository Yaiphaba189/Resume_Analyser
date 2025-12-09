import { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function FileUpload({ onFileSelect }) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const removeFile = () => {
    setFile(null);
    onFileSelect(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      <AnimatePresence mode='wait'>
        {!file ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current.click()}
            className={`
              relative group cursor-pointer
              flex flex-col items-center justify-center
              w-full h-64 rounded-2xl
              border-2 border-dashed transition-all duration-300
              ${dragActive 
                ? "border-green-500 bg-green-500/10" 
                : "border-zinc-800 bg-zinc-900/50 hover:border-green-500/50 hover:bg-zinc-900"
              }
            `}
          >
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
              onChange={handleChange}
            />
            
            <div className="mb-4 p-4 rounded-full bg-zinc-900 group-hover:scale-110 transition-transform duration-300 border border-zinc-800 group-hover:border-green-500/50">
              <Upload className={`w-8 h-8 ${dragActive ? "text-green-400" : "text-zinc-400 group-hover:text-green-500"}`} />
            </div>
            
            <p className="text-lg font-medium text-zinc-300 group-hover:text-white transition-colors">
              Click to upload or drag & drop
            </p>
            <p className="mt-2 text-sm text-zinc-500">
              PDF, DOCX, PNG, JPG up to 10MB
            </p>

            {dragActive && (
              <div className="absolute inset-0 rounded-2xl bg-green-500/5 pointer-events-none" />
            )}
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="flex items-center gap-4 p-4 w-full h-24 rounded-2xl bg-zinc-900 border border-green-500/20 shadow-[0_0_30px_-5px_rgba(34,197,94,0.1)]"
          >
            <div className="p-3 rounded-xl bg-green-500/10 text-green-500">
              <FileText className="w-8 h-8" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">
                {file.name}
              </p>
              <p className="text-xs text-zinc-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={removeFile}
              className="p-2 rounded-full hover:bg-red-500/10 text-zinc-500 hover:text-red-500 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="p-2 text-green-500">
              <CheckCircle className="w-6 h-6" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
