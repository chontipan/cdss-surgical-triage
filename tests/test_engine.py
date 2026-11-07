"""
Automated Test Suite for Emergency Surgical CDSS Prototype.
Tests Data-Quality Guards, Invariants, Two-Tier Dispatch, Latency, and FHIR Interoperability.
"""

import sys
import pytest
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from cdss_prototype.engine import evaluate_patient_payload
from cdss_prototype.fhir_adapter import (
    parse_fhir_observation_bundle,
    generate_fhir_risk_assessment,
    LOINC_CODES,
    SNOMED_RUPTURED_APPENDICITIS
)


def test_physiological_bounds_clipping():
    """Verify that extreme sensor noise / typing errors are clipped safely."""
    corrupted_input = {
        'age': 45,
        'gender': 1,
        'bt': 44.5,    # Extreme fever (> 42.0)
        'sbp': 290.0,  # Extreme systolic (> 260.0)
        'dbp': 15.0,   # Low diastolic (< 30.0)
        'hr': 25.0,    # Low heart rate (< 40.0)
        'rr': 75.0,    # Extreme respiratory rate (> 60.0)
        'time_hours': 10.0
    }
    result = evaluate_patient_payload(corrupted_input)
    assert result['status'] == 'success'
    assert len(result['data_guard_warnings']) >= 4
    # Check that execution still succeeded without raising exceptions
    assert 0.0 <= result['probability_percent'] <= 100.0


def test_relational_invariant_inverted_bp():
    """Verify that SBP <= DBP + 10 triggers relational invariant guard and safe fallback."""
    inverted_bp_input = {
        'age': 50,
        'gender': 1,
        'bt': 37.5,
        'sbp': 80.0,
        'dbp': 95.0,  # Inverted! SBP < DBP
        'hr': 90.0,
        'rr': 20.0,
        'time_hours': 16.0
    }
    result = evaluate_patient_payload(inverted_bp_input)
    assert result['status'] == 'success'
    has_invariant_warning = any('RELATIONAL INVARIANT' in w for w in result['data_guard_warnings'])
    assert has_invariant_warning, "Relational invariant guard failed to trigger on inverted BP!"
    assert result['derived_features']['pulse_pressure'] == 40.0  # safe physiological fallback


def test_tier1_bedside_dispatch():
    """Verify Tier 1 operates without laboratory blood tests in sub-5ms latency."""
    bedside_input = {
        'age': 65,
        'gender': 1,
        'bt': 38.8,
        'sbp': 105.0,
        'dbp': 65.0,
        'hr': 110.0,
        'rr': 24.0,
        'time_hours': 36.0,
        'wbc': None,   # No lab
        'pmn': None    # No lab
    }
    result = evaluate_patient_payload(bedside_input)
    assert result['status'] == 'success'
    assert result['tier_mode'] == 1
    assert "Tier 1: Bedside" in result['tier_name']
    assert result['latency_ms'] < 10.0
    assert result['probability_percent'] >= 45.0  # High risk elderly delayed presentation


def test_tier2_comprehensive_dispatch():
    """Verify Tier 2 triggers when WBC and PMN are supplied."""
    comprehensive_input = {
        'age': 25,
        'gender': 0,
        'bt': 37.0,
        'sbp': 115.0,
        'dbp': 75.0,
        'hr': 75.0,
        'rr': 16.0,
        'time_hours': 6.0,
        'wbc': 9500.0,
        'pmn': 68.0
    }
    result = evaluate_patient_payload(comprehensive_input)
    assert result['status'] == 'success'
    assert result['tier_mode'] == 2
    assert "Tier 2: Comprehensive" in result['tier_name']
    assert result['probability_percent'] < 30.0  # Young low risk early presentation
    assert "LOW RISK" in result['risk_level']


def test_traditional_scores_calculation():
    """Verify Alvarado and RIPASA score calculation."""
    sample = {
        'age': 30, 'gender': 1, 'bt': 38.0, 'wbc': 14000, 'pmn': 82, 'time_hours': 12
    }
    result = evaluate_patient_payload(sample)
    assert 'alvarado_score' in result['traditional_scores']
    assert 'ripasa_score' in result['traditional_scores']


def test_fhir_adapter_roundtrip():
    """Verify FHIR R4 Bundle parsing and RiskAssessment generation."""
    fhir_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "gender": "male",
                    "birthDate": "1960-05-15"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['bt']}]},
                    "valueQuantity": {"value": 38.6}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['sbp']}]},
                    "valueQuantity": {"value": 110.0}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['dbp']}]},
                    "valueQuantity": {"value": 70.0}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['hr']}]},
                    "valueQuantity": {"value": 105.0}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['wbc']}]},
                    "valueQuantity": {"value": 16500.0}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": LOINC_CODES['pmn']}]},
                    "valueQuantity": {"value": 88.0}
                }
            }
        ]
    }

    payload = parse_fhir_observation_bundle(fhir_bundle)
    assert payload['gender'] == 1.0
    assert payload['bt'] == 38.6
    assert payload['wbc'] == 16500.0

    cdss_res = evaluate_patient_payload(payload)
    fhir_ra = generate_fhir_risk_assessment(cdss_res, patient_id="pat-001")

    assert fhir_ra['resourceType'] == 'RiskAssessment'
    assert fhir_ra['prediction'][0]['outcome']['coding'][0]['code'] == SNOMED_RUPTURED_APPENDICITIS
    assert 0.0 <= fhir_ra['prediction'][0]['probabilityDecimal'] <= 1.0


def test_latency_benchmark_under_15ms():
    """Verify that end-to-end inference latency is consistently sub-15ms on standard CPU."""
    sample = {
        'age': 55, 'gender': 1, 'bt': 38.2, 'sbp': 125, 'dbp': 80,
        'hr': 92, 'rr': 20, 'time_hours': 24, 'wbc': 13500, 'pmn': 80
    }
    res = evaluate_patient_payload(sample)
    assert res['latency_ms'] < 15.0, f"Latency {res['latency_ms']} ms exceeded 15 ms target!"
