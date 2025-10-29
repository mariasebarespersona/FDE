# FDE

# Carrier Sales Assistant — Pitch & Negotiate (Metrics & Dashboard)

Minimal, production-minded metrics service and dashboard for the **Pitch & Negotiate** carrier workflow: FMCSA verification → load selection → rate negotiation → outcome logging → KPIs.

**Live API (health):** https://fde-production.up.railway.app/healthz  
**Repository:** https://github.com/mariasebarespersona/FDE

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [API](#api)
- [Quickstart (Local)](#quickstart-local)
- [Docker](#docker)
- [Deploy on Railway](#deploy-on-railway)
- [Environment Variables](#environment-variables)
- [HappyRobot Integration](#happyrobot-integration)
- [Dashboard](#dashboard)
- [Event Schema](#event-schema)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This repo contains:

- **FastAPI** service exposing:
  - `POST /events` — ingest structured call metrics
  - `GET /events` — list recent events (for the dashboard)
  - `GET /healthz` — machine-readable health probe
- **SQLite** storage (file path configurable)
- **Streamlit** dashboard that reads from the API and presents KPIs:
  - Total Calls, FMCSA Verified, Loads Pitched, Agreements, Win Rate
  - Agreed vs Loadboard (accepted only)
  - Sentiment & Rounds
  - Raw Events + Latest Call spotlight
- **Dockerfile** for reproducible builds and easy deployment (e.g., Railway)

---

## Architecture
HappyRobot Workflow
├─ FMCSA Verify ──┐
├─ Load Selected ──┼──▶ POST /events ──▶ FastAPI + SQLite ──▶ GET /events ──▶ Streamlit Dashboard
└─ Negotiate ──┘ (Railway) (local or hosted)


- Auth: `X-API-KEY` header on writes  
- HTTPS: provided by Railway  
- Health: `/healthz` for monitors and CI checks

---

## API

### POST `/events`
Ingest a single event (see [Event Schema](#event-schema)).

**Headers**



**Example**
```bash
curl -s -X POST https://<SERVICE>/events \
  -H 'Content-Type: application/json' \
  -H 'X-API-KEY: devkey123' \
  -d '{
    "run_id":"demo-123",
    "stage":"negotiate_result",
    "outcome":"accepted",
    "agreed_rate":1850,
    "loadboard_rate":2100,
    "negotiation_rounds":3,
    "equipment_type":"Flatbed",
    "origin":"Dallas",
    "destination":"NY"
  }'
```

GET /events

Query recent events for the dashboard. Supports limit (default 100).

/events?limit=100


Response: JSON array of event objects.

GET /healthz

Machine-readable health check:

{"ok": true, "ts": "2025-10-29T19:18:19.952431+00:00"}



**Quickstart**
# setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run API
export API_KEY=devkey123
export METRICS_DB=./metrics.db
uvicorn app:app --host 0.0.0.0 --port 8081 --reload

# (optional) run dashboard in another terminal
export METRICS_URL="http://localhost:8081"
streamlit run dashboard.py

**Docker**
docker build -t hr-metrics .
docker run --rm -p 8081:8081 \
  -e API_KEY=devkey123 \
  -e METRICS_DB=/tmp/metrics.db \
  hr-metrics

HappyRobot Integration

Add a Webhook (POST) node (using your reusable template) after each stage:

FMCSA Verify

stage = "fmcsa_verify"

outcome = "active" | "inactive" | "unknown"

carrier_mc (string), fmcsa_status (string)

Load Selected

stage = "load_selected"

outcome = "ready_to_pitch"

origin, destination, equipment_type, loadboard_rate (number), load_id (string)

Negotiate Result

stage = "negotiate_result"

outcome = "accepted" | "declined"

agreed_rate, model_offer, loadboard_rate, negotiation_rounds, transferred_to_rep (bool)

Webhook settings

URL: {{ METRICS_URL }}/events

Headers: Content-Type: application/json, X-API-KEY: {{ API_KEY }}

Preserve data types: ON (numbers remain numbers)

Define METRICS_URL and API_KEY in your HappyRobot Environment Variables.


**Dashboard**
export METRICS_URL="https://<SERVICE>.up.railway.app"
streamlit run dashboard.py
**EVENT SCHEMA**
export METRICS_URL="https://<SERVICE>.up.railway.app"
streamlit run dashboard.py

