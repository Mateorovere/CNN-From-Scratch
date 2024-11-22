import React, { useState } from "react";
import Canvas from "./components/Canvas";

const App = () => {
  const [prediction, setPrediction] = useState(null);

  const classifyImage = async (imageData) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error occurred while classifying");
    }
  };

  return (
    <div style={{ textAlign: "center", fontFamily: "Arial, sans-serif" }}>
      <h1>Handwritten Digit Classifier</h1>
      <p>Draw a digit on the canvas below and click "Classify"</p>
      <Canvas onClassify={classifyImage} />
      {prediction !== null && (
        <p style={{ fontSize: "24px", marginTop: "20px" }}>
          Predicted Digit: <strong>{prediction}</strong>
        </p>
      )}
    </div>
  );
};

export default App;
