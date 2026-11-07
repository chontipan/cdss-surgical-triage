# Clinical Decision Support System (CDSS) for Emergency Surgical Triage

### Architecture and Empirical Evaluation of a Data-Quality-Aware Clinical Decision Support System: Integrating LLM Priors and Machine Learning for Resource-Constrained Emergency Triage

## Research Project & Authors
Authors: **Sasithun Plengvittaya, MD** (Lampang Hospital) and **Chontipan Plengvittaya, PhD** (University of Phayao)

---

## 🌟 Live Interactive Web Demo
Access the interactive web prototype directly in your browser:  
👉 **[Live Web Application](https://chontipan.github.io/cdss-surgical-triage/)**

---

## 🚀 Key Software Engineering Highlights
- **Two-Tier Inference Decoupling:**
  - **Tier 1 (Bedside Zero-Lab Engine):** Executes in **< 2.0 ms** using non-invasive vital signs and symptom onset delay, requiring **zero blood tests** (0-minute laboratory turnaround delay).
  - **Tier 2 (Comprehensive Hybrid Engine):** Automatically triggers upon complete blood count ingestion, providing a calibrated consensus with **11.4 ms latency**.
- **Defensive Ingestion Guard:**
  - Enforces biological envelopes and cross-variable relational invariants ($SBP \ge DBP + 10$ mmHg) to protect models against human entry error and sensor noise.
- **Explainable AI (XAI):**
  - Instantaneous relative feature attribution breakdown for emergency clinicians.

---

## 💻 Local Quick Start (Zero External Dependencies)

### Windows 1-Click Launch:
Double-click `run_prototype.bat` or run:
```bash
python server.py
```
Your default web browser will automatically open: `http://localhost:8000`

---

## 📊 Empirical Cross-Validated Benchmarks (Cohort N = 443)
- **Alvarado Score (1986):** AUROC = **0.4603** (Fails on perforation)
- **RIPASA Score (2010):** AUROC = **0.5227** (Near random guess)
- **Deep MLP (Daldal 2021):** AUROC = **0.6796**
- **XGBoost (Chen 2016):** AUROC = **0.7141**
- **Random Forest (Akbulut 2020):** AUROC = **0.7256**
- **Proposed Tier 1 (Bedside Zero-Lab):** AUROC = **0.7351** (1.8 ms latency)
- **Proposed Tier 2 (Hybrid Consensus):** AUROC = **0.7531** (Apparent AUROC **0.8208**, 11.4 ms latency)

---

## 📄 License & Attribution
Open-source software released under the **MIT License**.
