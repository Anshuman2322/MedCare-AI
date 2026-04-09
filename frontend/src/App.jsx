import React, { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data.result);
      setConfidence(res.data.confidence);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="center-wrap font-sans">
      <div className="w-[95%] max-w-4xl p-10 glass-card relative">
        {/* Header: logo left, login right */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center text-white text-lg font-bold">MC</div>
            <div>
              <div className="text-sm text-cyan-200">MedCare-AI</div>
            </div>
          </div>

          <button className="border border-cyan-600 text-cyan-200 px-3 py-1 rounded-md hover:bg-white/5 transition">Login</button>
        </div>

        {/* Title */}
        <h1 className="text-4xl font-extrabold text-cyan-300 mb-6 text-center card-title">Precision AI Diagnosis</h1>

        {/* Upload box and actions in center */}
        <div className="flex flex-col items-center gap-6">
          <label className="dashed-upload flex flex-col items-center justify-center w-full max-w-2xl cursor-pointer">
            <svg className="cloud-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M7 18h10a4 4 0 0 0 0-8 5 5 0 0 0-9.9 1A3.5 3.5 0 0 0 7 18z" fill="currentColor" />
            </svg>
            <div className="text-cyan-100/80">Click to upload Chest X-Ray image</div>
            <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
          </label>

          <button
            onClick={handleUpload}
            disabled={loading}
            className="neon-btn"
          >
            <span className="text-lg">🔍</span>
            <span>{loading ? 'Analyzing...' : 'Upload & Analyze'}</span>
          </button>

          {preview && (
            <div className="w-full max-w-2xl">
              <img src={preview} alt="Preview" className="rounded-xl shadow-xl w-full object-cover" />
            </div>
          )}

          {result && (
            <div className="mt-6 p-5 bg-white/5 border border-cyan-700/10 rounded-2xl w-full max-w-2xl">
              <h2 className="text-xl font-semibold text-cyan-200 mb-2">Diagnosis Result</h2>
              <p className="text-gray-200 text-lg"><strong>Condition:</strong> {result}</p>
              {confidence && <p className="text-gray-400 mt-1"><strong>Confidence:</strong> {(confidence*100).toFixed(2)}%</p>}
            </div>
          )}
        </div>

        <footer className="mt-8 text-sm text-cyan-300 text-center">© 2025 MedCare-AI • Empowering Medical Intelligence</footer>
      </div>
    </div>
  );
}

export default App;
