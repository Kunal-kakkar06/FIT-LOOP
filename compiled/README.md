# FIT-LOOP Compiled — Integrated Model 1 + Wardrobe ML

This `compiled/` directory contains the fully integrated backend service and frontend dashboard combining **Model 1 (Body Measurement)** and **Wardrobe ML (Garment Detection)** from the FIT-LOOP project.

---

## 📁 Structure

```
compiled/
├── backend/
│   ├── main.py           # Unified FastAPI service (all endpoints)
│   └── requirements.txt  # Python dependencies
└── frontend/
    ├── src/              # React source (App.jsx, components/, api/)
    ├── dist/             # Production build output (ready to serve)
    ├── package.json
    ├── vite.config.js    # Dev server with API proxy → port 8000
    └── .env
```

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.10 with conda (`ml_env` recommended)
- Node.js ≥ 18 + npm
- YOLO weights at: `fitloop/models/wardrobe_ml/models/best.pt`

### 1. Start the Backend

```bash
cd compiled/backend

# Install deps (first time)
pip install -r requirements.txt

# Run the unified API server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Frontend (Dev Mode)

```bash
cd compiled/frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 3. Serve Production Build

```bash
# Serve the pre-built dist/ with any static file server
npx serve compiled/frontend/dist
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + model status |
| POST | `/api/v1/users` | Create a user |
| GET | `/api/v1/users` | List all users |
| POST | `/api/v1/measurements/estimate` | **Model 1** — body measurements from front/side images |
| POST | `/api/v1/scans` | **Wardrobe ML** — YOLO garment detection + color + embeddings |
| GET | `/api/v1/wardrobe/{user_id}` | Get user's wardrobe items |
| POST | `/api/v1/check-before-buy` | Duplicate detection before purchase |
| GET | `/api/v1/profile/{user_id}` | Integrated fit profile (measurements + wardrobe) |

---

## 🖥️ Frontend Tabs

| Tab | Description |
|-----|-------------|
| **Model 1 — Body Measurement** | Upload front + side photo to get body measurements |
| **Wardrobe Scanner** | Upload garment photo for YOLO detection + color analysis |
| **Check Before Buy** | Detect if a new garment duplicates your wardrobe |
| **Fit Passport** | Unified sizing dashboard across both models |

---

## ⚙️ Environment Notes

- Backend port: **8000**
- Frontend dev port: **5173** (Vite proxies `/api`, `/uploads`, `/health` → 8000)
- Database auto-created at `backend/fitloop.db` (SQLite, gitignored)
- Uploads stored at `backend/uploads/` (gitignored)

---

## 🔗 Related

- Full project: [FIT-LOOP GitHub](https://github.com/Kunal-kakkar06/FIT-LOOP)
- Model 1 source: `fitloop/models/model1_body_measurement/`
- Wardrobe ML source: `fitloop/models/wardrobe_ml/`
