import React, { useState } from "react";
import axios from "axios";

function ImageUpload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!image) return alert("Please select an image first!");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(`Prediction: ${res.data.prediction} | Confidence: ${res.data.confidence}`);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "40px" }}>
      <h2>🩺 MedCare AI - Medical Image Diagnosis</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      {preview && (
        <div>
          <img
            src={preview}
            alt="Preview"
            width="250"
            style={{ marginTop: 20, borderRadius: "10px" }}
          />
        </div>
      )}
      <br />
      <button
        onClick={handleUpload}
        style={{
          marginTop: 20,
          padding: "10px 20px",
          cursor: "pointer",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "6px",
        }}
      >
        Analyze Image
      </button>
      {result && <p style={{ marginTop: 20, fontWeight: "bold" }}>{result}</p>}
    </div>
  );
}

export default ImageUpload;
