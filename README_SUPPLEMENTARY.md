# Supplementary Materials: Clinical Decision Support System for Surgical Triage in Suspected Acute Appendicitis

**Authors:**
- Sasithun Plengvittaya, MD (Department of Surgery, Lampang Hospital, Lampang, Thailand)
- Chontipan Plengvittaya, PhD (School of Information and Communication Technology, University of Phayao, Phayao, Thailand - Corresponding Author: chontipan.pl@up.ac.th)

---

## 1. Archive Contents

This archive (`Supplementary_Materials_CDSS_Appendicitis.zip`) contains all reproducible code, de-identified datasets, benchmark evaluation tables, and pre-trained inference artifacts for peer-review and open scientific replication:

1. **`Supplementary_Materials.docx`**: Complete Supplementary Information document containing:
   - **Table S1**: Complete Clinical Feature Dictionary, LOINC / SNOMED CT Ontologies, and Reference Ranges.
   - **Table S2**: Extended Comparative Benchmark Performance Across 10 Metrics for All 10 Models (Lampang Cohort, N=443).
   - **Table S3**: External Cross-Continental Cohort Validation Breakdown (Regensburg Pediatric Cohort, Germany, N=165).
   - **Table S4**: Hyperparameter Optimization Grids, Regularization Schemes, and Search Spaces.
   - **Section S1**: HL7 FHIR R4 Bundle Ingestion and RiskAssessment Resource Representation.
   - **Section S2**: Zero-Leakage Data Provenance & Variable Exclusion Protocol.
   - **Section S3**: Standalone Reproduction Protocol.

2. **Datasets (CSV)**:
   - `Supplementary_Data1_Lampang_Cohort_N443.csv`: Primary de-identified Lampang Hospital research dataset (N = 443; 153 ruptured / 290 non-ruptured).
   - `Supplementary_Data2_Regensburg_Cohort_N165.csv`: External validation cohort benchmark (Regensburg Pediatric Cohort, Germany; N = 165 surgical cases; 51 complicated).
   - `Supplementary_Data3_Model_Benchmark_Detailed.csv`: Detailed performance metrics (AUROC, AUPRC, Accuracy, Balanced Accuracy, Sensitivity, Specificity, PPV, NPV, F1, Brier Score, Latency) across 10 algorithmic configurations.

3. **Reproducibility Scripts & Core Engine**:
   - `reproduce_benchmarks.py`: One-command verification script to validate datasets, Alvarado scores, and benchmark summary tables.
   - `cdss_engine/engine.py`: CDSS two-tier inference dispatcher and WSES guideline prior integration.
   - `cdss_engine/fhir_adapter.py`: HL7 FHIR R4 ingestion and RiskAssessment exporter.
   - `cdss_engine/models/`: Serialized joblib pipelines for Tier 1 and Tier 2 inference.
   - `cdss_engine/tests/test_engine.py`: Automated pytest test suite verifying zero-leakage, latency, and consensus behavior.

4. **Publication-Ready Figures (300 DPI)**:
   - `figures/figure1_system_architecture.png`
   - `figures/figure2_roc_curves.png`
   - `figures/figure3_feature_importance.png`
   - `figures/figure4_decision_curve.png`
   - `figures/figure5_calibration_curve.png`
   - `figures/figure6_shap_summary.png`

---

## 2. Quickstart Replication Instructions

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install numpy pandas scikit-learn joblib pytest
  ```

### Step 1: Run Benchmark Verification
```bash
python reproduce_benchmarks.py
```

### Step 2: Run Unit Tests
```bash
pytest cdss_engine/tests/ -v
```

---

## 3. Live Online Demo & Public Repository
- **Interactive Web Application:** https://chontipan.github.io/cdss-surgical-triage/
- **GitHub Repository:** https://github.com/chontipan/cdss-surgical-triage
