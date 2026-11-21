import { useEffect, useState } from "react";
import { Leaf, UploadCloud, ScanSearch, ShieldCheck, History, Sprout, AlertTriangle, Activity } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);

  async function refresh() {
    try {
      const [h, r] = await Promise.all([
        fetch(`${API}/api/health`),
        fetch(`${API}/api/history`)
      ]);
      setHealth(await h.json());
      setHistory(await r.json());
    } catch {
      setHealth({ model_ready: false, message: "Backend is not reachable." });
    }
  }

  useEffect(() => { refresh(); }, []);

  function chooseFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) {
      setError("Please select a JPG, PNG, or WEBP image.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("image", file);
    try {
      const response = await fetch(`${API}/api/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");
      setResult(data);
      await refresh();
    } catch (e) {
      setError(e.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-icon"><Leaf size={23}/></span><span>AgriGuard <b>AI</b></span></a>
        <nav><a href="#detect">Disease detection</a><a href="#how">How it works</a><a href="#history">History</a></nav>
        <span className={`status ${health?.model_ready ? "ready" : "not-ready"}`}><i/> {health?.model_ready ? "Model online" : "Model not trained"}</span>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><Sprout size={15}/> SMART AGRICULTURE · EARLY SCREENING</div>
            <h1>Healthier crops start with <em>early detection.</em></h1>
            <p>Upload a crop leaf image to screen for visible disease symptoms using an AI image-classification model.</p>
            <a className="primary-link" href="#detect">Analyze a crop image <ScanSearch size={17}/></a>
            <div className="trust-note"><ShieldCheck size={17}/> Decision-support prototype · Expert confirmation recommended</div>
          </div>
          <div className="hero-art">
            <div className="art-circle"><Leaf size={112} strokeWidth={1.1}/></div>
            <div className="floating-card"><span className="dot"/> Crop health assistant <b>AI screening</b></div>
          </div>
        </section>

        <section className="metrics">
          <div><span className="metric-icon"><ScanSearch/></span><b>Image-based</b><small>Visual symptom screening</small></div>
          <div><span className="metric-icon"><Activity/></span><b>Model confidence</b><small>Top predicted classes</small></div>
          <div><span className="metric-icon"><ShieldCheck/></span><b>Responsible guidance</b><small>Expert confirmation encouraged</small></div>
        </section>

        <section className="section" id="detect">
          <div className="section-heading"><div><div className="eyebrow">AI DIAGNOSTIC TOOL</div><h2>Check your crop health</h2><p>Use a clear, close-up image of one leaf with good lighting.</p></div></div>
          <div className="workspace">
            <div className="upload-panel">
              <div className="panel-title"><span>01</span><div><b>Upload crop image</b><small>JPG, PNG or WEBP · Maximum 8 MB</small></div></div>
              <label className="dropzone">
                {preview ? <img src={preview} alt="Selected crop preview"/> : <div className="drop-content"><UploadCloud size={38}/><b>Choose a leaf image</b><span>Browse from your device</span></div>}
                <input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile}/>
              </label>
              <button className="analyze-btn" disabled={!file || loading} onClick={analyze}>{loading ? "Analyzing image..." : "Analyze image"} <ScanSearch size={18}/></button>
              {error && <div className="error"><AlertTriangle size={17}/>{error}</div>}
              {!health?.model_ready && <p className="hint">The model is not trained yet. Follow the project README to download data and train it before analyzing images.</p>}
            </div>

            <div className="result-panel">
              <div className="panel-title"><span>02</span><div><b>Analysis result</b><small>AI prediction and suggested next steps</small></div></div>
              {!result ? <div className="empty-result"><div className="empty-icon"><Leaf size={34}/></div><b>Your result will appear here</b><p>Upload an image and select Analyze image to begin.</p></div> :
                <div className="result-content">
                  <div className={`result-status ${result.status === "Healthy" ? "healthy" : "alert"}`}>{result.status}</div>
                  <h3>{result.condition}</h3>
                  <p className="crop-name">Crop: {result.crop}</p>
                  <div className="confidence"><div><span>Model confidence</span><b>{result.confidence}%</b></div><div className="bar"><i style={{width: `${result.confidence}%`}}/></div></div>
                  <h4>Top predictions</h4>
                  {result.top_predictions.map(p => <div className="prediction" key={p.label}><span>{p.label.replaceAll("___"," – ").replaceAll("_"," ")}</span><b>{p.confidence}%</b></div>)}
                  <h4>Suggested next steps</h4>
                  <ul>{result.advice.map((a,i)=><li key={i}>{a}</li>)}</ul>
                  <p className="disclaimer">This is an AI screening result, not a confirmed diagnosis. Confidence is not the probability that a disease is present. Do not choose or apply pesticides based only on this output.</p>
                </div>
              }
            </div>
          </div>
        </section>

        <section className="how-section" id="how">
          <div className="eyebrow">SIMPLE WORKFLOW</div><h2>How AgriGuard AI works</h2>
          <div className="steps">
            <div><span>1</span><UploadCloud/><b>Upload</b><p>Select a clear crop leaf photograph.</p></div>
            <div><span>2</span><ScanSearch/><b>Analyze</b><p>The trained image model predicts a class.</p></div>
            <div><span>3</span><Sprout/><b>Review</b><p>Review confidence and responsible next steps.</p></div>
          </div>
        </section>

        <section className="history-section" id="history">
          <div className="history-title"><div><div className="eyebrow">RECENT ACTIVITY</div><h2>Detection history</h2></div><button onClick={refresh}><History size={16}/> Refresh</button></div>
          {history.length === 0 ? <p className="history-empty">No predictions recorded yet.</p> : <div className="history-list">{history.slice(0,8).map(item=><div className="history-row" key={item.id}><span className="history-leaf"><Leaf size={18}/></span><div><b>{item.crop} · {item.condition}</b><small>{new Date(item.timestamp).toLocaleString()}</small></div><strong>{item.confidence}%</strong></div>)}</div>}
        </section>
      </main>
      <footer><a className="brand" href="#"><span className="brand-icon"><Leaf size={19}/></span>AgriGuard <b>AI</b></a><span>Academic project prototype · CSE471</span><span>AI-assisted screening, not a substitute for expert diagnosis.</span></footer>
    </div>
  );
}
