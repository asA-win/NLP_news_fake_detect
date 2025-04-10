import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleCheck = async () => {
    if (!text.trim()) {
      setError("Please enter some text.");
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:5000/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError('Oops! Something went wrong. Please try again.');
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <header>
        <h1>🕵️ AI Fact Checker</h1>
      </header>

      <main>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a news article or claim here..."
          rows="10"
        />

        {error && <p className="error">{error}</p>}

        <button onClick={handleCheck} disabled={loading}>
          {loading ? <span className="spinner" /> : 'Check Facts'}
        </button>

        <div className="results">
          {results.map((item, index) => (
            <div className="card" key={index}>
              <h3>Claim:</h3>
              <p>{item.claim}</p>
              <p><strong>🔍 Label:</strong> {item.label} (Confidence: {item.score})</p>
              <p><strong>📚 Evidence ({item.source}):</strong> {item.evidence}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

export default App;
