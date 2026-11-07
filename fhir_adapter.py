"""
HL7 FHIR R4 Adapter for Emergency Surgical CDSS.
Provides interoperability mapping between standard FHIR resources
(Patient, Observation, Bundle) and the Two-Tier CDSS Engine.
Emits FHIR RiskAssessment resources with standardized LOINC and SNOMED-CT codings.
"""

import uuid
import datetime

try:
    from engine import evaluate_patient_payload
except ImportError:
    from cdss_prototype.engine import evaluate_patient_payload

# Standard Clinical Coding Systems
LOINC_CODES = {
    'bt': '8310-5',        # Body temperature
    'sbp': '8480-6',       # Systolic blood pressure
    'dbp': '8462-4',       # Diastolic blood pressure
    'hr': '8867-4',        # Heart rate
    'rr': '9279-1',        # Respiratory rate
    'wbc': '6690-2',       # Leukocytes count in blood
    'pmn': '770-8',        # Neutrophils/100 leukocytes
    'pain_time': '69741-7' # Duration of pain
}

SNOMED_RUPTURED_APPENDICITIS = '70650003' # Rupture of appendix (disorder)


def parse_fhir_observation_bundle(bundle_json: dict) -> dict:
    """
    Parses a FHIR R4 Bundle containing Patient and Observation resources
    into a standardized raw patient telemetry dictionary for the CDSS engine.
    """
    payload = {
        'age': 35.0,
        'gender': 1.0,
        'bt': 37.0,
        'sbp': 120.0,
        'dbp': 80.0,
        'hr': 80.0,
        'rr': 20.0,
        'time_hours': 12.0,
        'wbc': None,
        'pmn': None
    }

    entries = bundle_json.get('entry', [])
    for entry in entries:
        resource = entry.get('resource', {})
        res_type = resource.get('resourceType')

        if res_type == 'Patient':
            gender_str = resource.get('gender', 'male')
            payload['gender'] = 1.0 if gender_str == 'male' else 0.0
            birth_date = resource.get('birthDate')
            if birth_date:
                try:
                    birth_year = int(birth_date.split('-')[0])
                    current_year = datetime.datetime.now().year
                    payload['age'] = float(current_year - birth_year)
                except Exception:
                    pass

        elif res_type == 'Observation':
            code_entries = resource.get('code', {}).get('coding', [])
            val = resource.get('valueQuantity', {}).get('value')
            if val is None:
                continue

            for coding in code_entries:
                code = coding.get('code')
                if code == LOINC_CODES['bt']:
                    payload['bt'] = float(val)
                elif code == LOINC_CODES['sbp']:
                    payload['sbp'] = float(val)
                elif code == LOINC_CODES['dbp']:
                    payload['dbp'] = float(val)
                elif code == LOINC_CODES['hr']:
                    payload['hr'] = float(val)
                elif code == LOINC_CODES['rr']:
                    payload['rr'] = float(val)
                elif code == LOINC_CODES['wbc']:
                    payload['wbc'] = float(val)
                elif code == LOINC_CODES['pmn']:
                    payload['pmn'] = float(val)
                elif code == LOINC_CODES['pain_time']:
                    payload['time_hours'] = float(val)

    return payload


def generate_fhir_risk_assessment(cdss_result: dict, patient_id: str = "example-patient") -> dict:
    """
    Transforms the CDSS evaluation result into a standardized FHIR R4 RiskAssessment resource.
    """
    assessment_id = f"ra-{uuid.uuid4().hex[:8]}"
    prob_fraction = round(cdss_result['probability_percent'] / 100.0, 4)

    fhir_ra = {
        "resourceType": "RiskAssessment",
        "id": assessment_id,
        "status": "final",
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "occurrenceDateTime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": {
            "coding": [
                {
                    "system": "https://github.com/chontipan/cdss-surgical-triage",
                    "code": cdss_result['tier_name'].replace(" ", "-"),
                    "display": cdss_result['tier_name']
                }
            ],
            "text": "Data-Quality-Aware Two-Tier Machine Learning Inference Pipeline"
        },
        "basis": [
            {
                "display": f"Derived Shock Index: {cdss_result['derived_features']['shock_index']}, Pulse Pressure: {cdss_result['derived_features']['pulse_pressure']} mmHg"
            }
        ],
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": SNOMED_RUPTURED_APPENDICITIS,
                            "display": "Rupture of appendix"
                        }
                    ],
                    "text": "Appendiceal Perforation / Gangrenous Rupture"
                },
                "probabilityDecimal": prob_fraction,
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": "high" if prob_fraction >= 0.45 else ("moderate" if prob_fraction >= 0.22 else "low"),
                            "display": cdss_result['risk_level']
                        }
                    ]
                },
                "rationale": cdss_result['recommendation']
            }
        ],
        "note": [
            {
                "text": f"Execution latency: {cdss_result['latency_ms']} ms. Quality guard warnings: {len(cdss_result['data_guard_warnings'])}"
            }
        ]
    }
    return fhir_ra
