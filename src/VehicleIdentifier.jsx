import { useState, useRef, useCallback } from "react";
import "./VehicleIdentifier.css";

const API_KEY = "";

const VEHICLE_PROMPT = `You are an expert automotive analyst and car enthusiast with encyclopedic knowledge of vehicles worldwide.
Analyze this image or video frame carefully and identify everything about the vehicle(s) shown.
Look for body shape, grille design, headlight/taillight shapes, badge/logo, wheel design, interior details if visible, license plate region if visible, any text or emblems.
Return ONLY a valid JSON object with this exact structure:
{"make":"Manufacturer name","model":"Model name","variant":"Trim or variant if detectable","year_range":"Estimated year or range","body_type":"Sedan/SUV/Coupe/Hatchback/Pickup/Van/Convertible/Wagon/Sports Car/Supercar/Electric","segment":"Market segment","drivetrain":"FWD/RWD/AWD/4WD/Unknown","powertrain":"ICE/Hybrid/Plug-in Hybrid/Electric/Unknown","engine_estimate":"Engine or motor info","color":"Exterior color","country_of_origin":"Brand origin country","notable_features":["feature1","feature2"],"description":"3-4 sentence expert analysis.","confidence":85}`;

const POWERTRAIN_COLORS = {
  Electric: "#22d3ee",
  "Plug-in Hybrid": "#a78bfa",
  Hybrid: "#34d399",
  ICE: "#f97316",
  Unknown: "#6b7280",
};

const BODY_EMOJI = {
  Sedan: "🚗",
  SUV: "🚙",
  Coupe: "🏎️",
  Hatchback: "🚘",
  Pickup: "🛻",
  Van: "🚐",
  Convertible: "🏎️",
  Wagon: "🚗",
  "Sports Car": "🏎️",
  Supercar: "🏎️",
  Electric: "⚡",
};

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };

    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function extractVideoFrame(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    const canvas = document.createElement("canvas");

    video.src = URL.createObjectURL(file);
    video.muted = true;

    video.onloadeddata = () => {
      video.currentTime = Math.min(1.5, video.duration * 0.2);
    };

    video.onseeked = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      const base64 = canvas.toDataURL("image/jpeg", 0.92).split(",")[1];

      URL.revokeObjectURL(video.src);
      resolve(base64);
    };

    video.onerror = reject;
  });
}

async function callGemini(base64Data, mimeType) {
  const response = await fetch(
     `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${API_KEY}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                inline_data: {
                  mime_type: mimeType,
                  data: base64Data,
                },
              },
              {
                text: VEHICLE_PROMPT,
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 1000,
        },
      }),
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error?.message || `API error ${response.status}`);
  }

  const data = await response.json();
  const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const cleaned = raw.replace(/```json|```/g, "").trim();

  return JSON.parse(cleaned);
}

function Badge({ label, color, background }) {
  return (
    <span
      className="badge"
      style={{
        color,
        background,
        border: `1px solid ${color}44`,
      }}
    >
      {label}
    </span>
  );
}

function StatBox({ label, value, accent }) {
  return (
    <div className="stat-box" style={{ borderTop: `2px solid ${accent}` }}>
      <div className="stat-label">{label}</div>
      <div className="stat-val">{value || "—"}</div>
    </div>
  );
}

export default function VehicleIdentifier() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [frameUrl, setFrameUrl] = useState("");
  const [isVideo, setIsVideo] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const inputRef = useRef(null);

  const handleFile = useCallback((selectedFile) => {
    if (!selectedFile) return;

    const videos = ["video/mp4", "video/webm", "video/quicktime"];
    const images = ["image/jpeg", "image/png", "image/webp", "image/gif"];

    if (![...videos, ...images].includes(selectedFile.type)) {
      setError("Supported: JPG, PNG, WEBP, GIF, MP4, WebM, MOV");
      return;
    }

    setFile(selectedFile);
    setIsVideo(videos.includes(selectedFile.type));
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setFrameUrl("");
    setResult(null);
    setError("");
  }, []);

  const reset = () => {
    setFile(null);
    setPreviewUrl("");
    setFrameUrl("");
    setIsVideo(false);
    setResult(null);
    setError("");
    setLoading(false);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      let base64;
      let mimeType;
      let analyzedFrameUrl;

      if (isVideo) {
        setLoadingText("Extracting best frame…");
        base64 = await extractVideoFrame(file);
        mimeType = "image/jpeg";
        analyzedFrameUrl = `data:image/jpeg;base64,${base64}`;
      } else {
        setLoadingText("Reading image…");
        base64 = await fileToBase64(file);
        mimeType = file.type;
        analyzedFrameUrl = previewUrl;
      }

      setFrameUrl(analyzedFrameUrl);

      setLoadingText("Running visual analysis…");
      const vehicleData = await callGemini(base64, mimeType);

      setLoadingText("Compiling vehicle report…");
      setResult(vehicleData);
    } catch (err) {
      setError(err.message || "Analysis failed. Check your API key.");
    } finally {
      setLoading(false);
      setLoadingText("");
    }
  };

  const confidence = Number(result?.confidence || 0);
  const confidenceColor =
    confidence >= 75 ? "#22c55e" : confidence >= 50 ? "#f59e0b" : "#ef4444";

  const powertrainColor =
    POWERTRAIN_COLORS[result?.powertrain] || POWERTRAIN_COLORS.Unknown;

  const bodyEmoji = BODY_EMOJI[result?.body_type] || "🚗";

  return (
    <div className="page">
      <div className="wrap">
        <div className="header">
          <div className="pill">
            <span className="dot"></span>
            <span>Powered by Gemini Vision</span>
          </div>

          <h1>Vehicle Identifier</h1>

          <p className="subtitle">
            Drop any car photo or video · AI identifies make, model, year,
            specs &amp; more
          </p>
        </div>

        {!file ? (
          <div
            className={`dropzone ${dragging ? "drag" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFile(e.dataTransfer.files[0]);
            }}
          >
            <div className="icon">🚗</div>
            <p>Drop a car photo or video</p>
            <small>JPG · PNG · WEBP · MP4 · WebM · MOV</small>
            <div className="browse-btn">Browse files</div>

            <input
              ref={inputRef}
              type="file"
              hidden
              accept="image/*,video/mp4,video/webm,video/quicktime"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        ) : (
          <div className="preview-wrap">
            {isVideo ? (
              <video src={previewUrl} controls />
            ) : (
              <img src={previewUrl} alt="Preview" />
            )}

            <button className="remove-btn" onClick={reset}>
              ✕ Remove
            </button>

            {isVideo && (
              <div className="vid-label">
                📽 Video — best frame will be extracted
              </div>
            )}
          </div>
        )}

        <button
          className="analyze-btn"
          onClick={analyze}
          disabled={!file || loading}
        >
          {loading ? `⏳ ${loadingText}` : "🔍 Identify Vehicle"}
        </button>

        {error && <div className="error-box">⚠️ {error}</div>}

        {result && (
          <div className="result">
            <div className="card-header">
              <div className="card-top">
                <div>
                  <div className="card-badges">
                    <span className="emoji">{bodyEmoji}</span>

                    <Badge
                      label={result.powertrain || "Unknown"}
                      color={powertrainColor}
                      background={`${powertrainColor}22`}
                    />

                    {result.drivetrain && result.drivetrain !== "Unknown" && (
                      <Badge
                        label={result.drivetrain}
                        color="#94a3b8"
                        background="#1e293b"
                      />
                    )}
                  </div>

                  <div className="car-name">
                    {result.make} {result.model}
                  </div>

                  {result.variant && (
                    <div className="car-variant">{result.variant}</div>
                  )}
                </div>

                <div>
                  <div className="confidence-label">Confidence</div>
                  <div
                    className="confidence-val"
                    style={{ color: confidenceColor }}
                  >
                    {confidence}%
                  </div>
                </div>
              </div>

              <div className="card-tags">
                {result.year_range && (
                  <Badge
                    label={result.year_range}
                    color="#c4b5fd"
                    background="#1e1b4b"
                  />
                )}

                {result.body_type && (
                  <Badge
                    label={result.body_type}
                    color="#7dd3fc"
                    background="#0c2a3e"
                  />
                )}

                {result.segment && (
                  <Badge
                    label={result.segment}
                    color="#86efac"
                    background="#052e16"
                  />
                )}

                {result.country_of_origin && (
                  <Badge
                    label={`🌍 ${result.country_of_origin}`}
                    color="#fcd34d"
                    background="#1c1200"
                  />
                )}
              </div>
            </div>

            {frameUrl && (
              <div className="frame-strip">
                <img src={frameUrl} alt="Analyzed frame" />
                <div className="frame-overlay"></div>
                <div className="frame-label">ANALYZED FRAME</div>

                {result.powertrain === "Electric" && (
                  <div className="ev-label">⚡ ELECTRIC VEHICLE</div>
                )}
              </div>
            )}

            <div className="card-body">
              <div className="stats-grid">
                <StatBox label="Color" value={result.color} accent="#6366f1" />

                <StatBox
                  label="Powertrain"
                  value={result.engine_estimate}
                  accent={powertrainColor}
                />

                <StatBox
                  label="Drive"
                  value={result.drivetrain}
                  accent="#38bdf8"
                />

                <StatBox
                  label="Origin"
                  value={result.country_of_origin}
                  accent="#fbbf24"
                />
              </div>

              <div className="description">
                <p>{result.description}</p>
              </div>

              {result.notable_features?.length > 0 && (
                <div>
                  <div className="features-label">Notable features</div>

                  <div className="features-list">
                    {result.notable_features.map((feature, index) => (
                      <span className="feature-tag" key={index}>
                        ✦ {feature}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <p className="footer">
          For production use, proxy Gemini calls through your own backend to
          protect your API key.
        </p>
      </div>
    </div>
  );
}