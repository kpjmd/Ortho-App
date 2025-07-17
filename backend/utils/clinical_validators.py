"""
Clinical Validators for Healthcare-Specific Data Validation
Provides specialized validation rules for orthopedic recovery data
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
from enum import Enum
import numpy as np


class ClinicalSeverity(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class OrthopedicCondition(str, Enum):
    ACL_TEAR = "acl_tear"
    MENISCUS_TEAR = "meniscus_tear"
    CARTILAGE_DEFECT = "cartilage_defect"
    KNEE_OSTEOARTHRITIS = "knee_osteoarthritis"
    TOTAL_KNEE_REPLACEMENT = "total_knee_replacement"
    ROTATOR_CUFF_TEAR = "rotator_cuff_tear"
    LABRAL_TEAR = "labral_tear"
    SHOULDER_INSTABILITY = "shoulder_instability"
    SHOULDER_OSTEOARTHRITIS = "shoulder_osteoarthritis"
    TOTAL_SHOULDER_REPLACEMENT = "total_shoulder_replacement"


class ClinicalValidators:
    """
    Healthcare-specific validation rules for orthopedic recovery data
    Implements evidence-based thresholds and clinical guidelines
    """
    
    def __init__(self):
        # Evidence-based clinical thresholds
        self.clinical_thresholds = {
            'heart_rate': {
                'resting': {
                    'normal': (60, 100),
                    'bradycardia': (40, 60),
                    'tachycardia': (100, 180)
                },
                'exercise': {
                    'light': (90, 126),
                    'moderate': (127, 153),
                    'vigorous': (154, 180),
                    'maximum': (181, 220)
                }
            },
            'blood_pressure': {
                'systolic': {
                    'normal': (90, 120),
                    'elevated': (120, 129),
                    'stage1': (130, 139),
                    'stage2': (140, 180),
                    'crisis': (180, 250)
                },
                'diastolic': {
                    'normal': (60, 80),
                    'elevated': (80, 89),
                    'stage1': (90, 99),
                    'stage2': (100, 120),
                    'crisis': (120, 150)
                }
            },
            'oxygen_saturation': {
                'normal': (95, 100),
                'mild_hypoxemia': (90, 94),
                'moderate_hypoxemia': (85, 89),
                'severe_hypoxemia': (80, 84),
                'critical': (0, 79)
            },
            'sleep': {
                'total_duration': {
                    'recommended': (7, 9),
                    'acceptable': (6, 10),
                    'concerning': (4, 6),
                    'critical': (0, 4)
                },
                'deep_sleep_percentage': {
                    'normal': (20, 30),
                    'low': (10, 20),
                    'very_low': (0, 10)
                },
                'rem_sleep_percentage': {
                    'normal': (20, 25),
                    'low': (10, 20),
                    'very_low': (0, 10)
                }
            },
            'activity': {
                'steps': {
                    'sedentary': (0, 2500),
                    'low_active': (2500, 5000),
                    'somewhat_active': (5000, 7500),
                    'active': (7500, 10000),
                    'highly_active': (10000, 50000)
                },
                'distance_km': {
                    'sedentary': (0, 2),
                    'low_active': (2, 4),
                    'somewhat_active': (4, 6),
                    'active': (6, 8),
                    'highly_active': (8, 50)
                }
            }
        }
        
        # Condition-specific recovery thresholds
        self.recovery_thresholds = {
            OrthopedicCondition.ACL_TEAR: {
                'early_recovery': {
                    'steps_per_day': (0, 3000),
                    'max_walking_speed': 3.0,  # km/h
                    'expected_pain_level': (3, 7)
                },
                'mid_recovery': {
                    'steps_per_day': (3000, 6000),
                    'max_walking_speed': 4.0,
                    'expected_pain_level': (2, 5)
                },
                'late_recovery': {
                    'steps_per_day': (6000, 12000),
                    'max_walking_speed': 5.0,
                    'expected_pain_level': (0, 3)
                }
            },
            OrthopedicCondition.TOTAL_KNEE_REPLACEMENT: {
                'early_recovery': {
                    'steps_per_day': (0, 2000),
                    'max_walking_speed': 2.0,
                    'expected_pain_level': (4, 8)
                },
                'mid_recovery': {
                    'steps_per_day': (2000, 5000),
                    'max_walking_speed': 3.5,
                    'expected_pain_level': (2, 6)
                },
                'late_recovery': {
                    'steps_per_day': (5000, 10000),
                    'max_walking_speed': 4.5,
                    'expected_pain_level': (0, 4)
                }
            },
            OrthopedicCondition.ROTATOR_CUFF_TEAR: {
                'early_recovery': {
                    'steps_per_day': (2000, 8000),
                    'max_walking_speed': 4.0,
                    'expected_pain_level': (3, 7),
                    'arm_elevation_concern': True
                },
                'mid_recovery': {
                    'steps_per_day': (4000, 10000),
                    'max_walking_speed': 5.0,
                    'expected_pain_level': (2, 5),
                    'arm_elevation_concern': True
                },
                'late_recovery': {
                    'steps_per_day': (6000, 15000),
                    'max_walking_speed': 5.5,
                    'expected_pain_level': (0, 3),
                    'arm_elevation_concern': False
                }
            }
        }
        
        # Age-based adjustments
        self.age_adjustments = {
            'heart_rate_max': lambda age: 220 - age,
            'target_heart_rate_lower': lambda age: (220 - age) * 0.5,
            'target_heart_rate_upper': lambda age: (220 - age) * 0.85,
            'steps_adjustment': {
                (18, 30): 1.0,
                (31, 50): 0.9,
                (51, 65): 0.8,
                (66, 80): 0.7,
                (81, 100): 0.6
            }
        }
    
    def validate_physiological_parameters(self, data: Dict[str, Any], 
                                        patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Validate physiological parameters against clinical standards
        
        Args:
            data: Wearable data dictionary
            patient_context: Patient context (age, diagnosis, recovery_stage)
            
        Returns:
            List of validation results
        """
        results = []
        
        # Heart rate validation
        if 'heart_rate' in data:
            hr_results = self._validate_heart_rate(data['heart_rate'], patient_context)
            results.extend(hr_results)
        
        # Blood pressure validation
        if 'blood_pressure' in data:
            bp_results = self._validate_blood_pressure(data['blood_pressure'], patient_context)
            results.extend(bp_results)
        
        # Oxygen saturation validation
        if 'oxygen_saturation' in data:
            o2_results = self._validate_oxygen_saturation(data['oxygen_saturation'], patient_context)
            results.extend(o2_results)
        
        # Sleep validation
        if 'sleep_data' in data:
            sleep_results = self._validate_sleep_patterns(data['sleep_data'], patient_context)
            results.extend(sleep_results)
        
        # Activity validation
        if 'activity_data' in data:
            activity_results = self._validate_activity_levels(data['activity_data'], patient_context)
            results.extend(activity_results)
        
        return results
    
    def _validate_heart_rate(self, hr_data: Dict[str, Any], 
                           patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate heart rate data against clinical standards"""
        results = []
        
        age = patient_context.get('age', 40) if patient_context else 40
        max_hr = self.age_adjustments['heart_rate_max'](age)
        
        # Resting heart rate
        if 'resting_hr' in hr_data:
            resting_hr = hr_data['resting_hr']
            resting_range = self.clinical_thresholds['heart_rate']['resting']['normal']
            
            if resting_hr < resting_range[0]:
                results.append({
                    'parameter': 'resting_heart_rate',
                    'value': resting_hr,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Bradycardia detected: {resting_hr} bpm (normal: {resting_range[0]}-{resting_range[1]})',
                    'clinical_significance': 'May indicate cardiac conduction issues or excellent fitness'
                })
            elif resting_hr > resting_range[1]:
                results.append({
                    'parameter': 'resting_heart_rate',
                    'value': resting_hr,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Tachycardia detected: {resting_hr} bpm (normal: {resting_range[0]}-{resting_range[1]})',
                    'clinical_significance': 'May indicate stress, dehydration, or cardiac issues'
                })
        
        # Maximum heart rate
        if 'max_hr' in hr_data:
            max_hr_recorded = hr_data['max_hr']
            
            if max_hr_recorded > max_hr:
                results.append({
                    'parameter': 'maximum_heart_rate',
                    'value': max_hr_recorded,
                    'severity': ClinicalSeverity.SEVERE,
                    'message': f'Heart rate {max_hr_recorded} exceeds age-predicted maximum {max_hr}',
                    'clinical_significance': 'Requires immediate clinical review'
                })
        
        return results
    
    def _validate_blood_pressure(self, bp_data: Dict[str, Any], 
                               patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate blood pressure against clinical guidelines"""
        results = []
        
        if 'systolic' in bp_data and 'diastolic' in bp_data:
            systolic = bp_data['systolic']
            diastolic = bp_data['diastolic']
            
            # Systolic validation
            if systolic >= 180:
                results.append({
                    'parameter': 'systolic_blood_pressure',
                    'value': systolic,
                    'severity': ClinicalSeverity.CRITICAL,
                    'message': f'Hypertensive crisis: {systolic} mmHg',
                    'clinical_significance': 'Requires immediate medical attention'
                })
            elif systolic >= 140:
                results.append({
                    'parameter': 'systolic_blood_pressure',
                    'value': systolic,
                    'severity': ClinicalSeverity.SEVERE,
                    'message': f'Stage 2 hypertension: {systolic} mmHg',
                    'clinical_significance': 'Requires medical management'
                })
            elif systolic >= 130:
                results.append({
                    'parameter': 'systolic_blood_pressure',
                    'value': systolic,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Stage 1 hypertension: {systolic} mmHg',
                    'clinical_significance': 'Lifestyle modifications recommended'
                })
            
            # Diastolic validation
            if diastolic >= 120:
                results.append({
                    'parameter': 'diastolic_blood_pressure',
                    'value': diastolic,
                    'severity': ClinicalSeverity.CRITICAL,
                    'message': f'Hypertensive crisis: {diastolic} mmHg',
                    'clinical_significance': 'Requires immediate medical attention'
                })
            elif diastolic >= 90:
                results.append({
                    'parameter': 'diastolic_blood_pressure',
                    'value': diastolic,
                    'severity': ClinicalSeverity.SEVERE,
                    'message': f'Stage 2 hypertension: {diastolic} mmHg',
                    'clinical_significance': 'Requires medical management'
                })
        
        return results
    
    def _validate_oxygen_saturation(self, o2_data: Dict[str, Any], 
                                  patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate oxygen saturation levels"""
        results = []
        
        if 'spo2' in o2_data:
            spo2 = o2_data['spo2']
            
            if spo2 < 85:
                results.append({
                    'parameter': 'oxygen_saturation',
                    'value': spo2,
                    'severity': ClinicalSeverity.CRITICAL,
                    'message': f'Severe hypoxemia: {spo2}%',
                    'clinical_significance': 'Requires immediate oxygen therapy'
                })
            elif spo2 < 90:
                results.append({
                    'parameter': 'oxygen_saturation',
                    'value': spo2,
                    'severity': ClinicalSeverity.SEVERE,
                    'message': f'Moderate hypoxemia: {spo2}%',
                    'clinical_significance': 'Requires medical evaluation'
                })
            elif spo2 < 95:
                results.append({
                    'parameter': 'oxygen_saturation',
                    'value': spo2,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Mild hypoxemia: {spo2}%',
                    'clinical_significance': 'Monitor closely, consider medical evaluation'
                })
        
        return results
    
    def _validate_sleep_patterns(self, sleep_data: Dict[str, Any], 
                               patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate sleep patterns against clinical standards"""
        results = []
        
        # Total sleep duration
        if 'total_sleep_hours' in sleep_data:
            sleep_hours = sleep_data['total_sleep_hours']
            
            if sleep_hours < 4:
                results.append({
                    'parameter': 'total_sleep_duration',
                    'value': sleep_hours,
                    'severity': ClinicalSeverity.SEVERE,
                    'message': f'Severe sleep deprivation: {sleep_hours} hours',
                    'clinical_significance': 'May impair recovery and increase injury risk'
                })
            elif sleep_hours < 6:
                results.append({
                    'parameter': 'total_sleep_duration',
                    'value': sleep_hours,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Insufficient sleep: {sleep_hours} hours',
                    'clinical_significance': 'May affect recovery and performance'
                })
        
        # Deep sleep percentage
        if 'deep_sleep_percentage' in sleep_data:
            deep_sleep = sleep_data['deep_sleep_percentage']
            
            if deep_sleep < 10:
                results.append({
                    'parameter': 'deep_sleep_percentage',
                    'value': deep_sleep,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Low deep sleep: {deep_sleep}%',
                    'clinical_significance': 'May impair physical recovery'
                })
        
        return results
    
    def _validate_activity_levels(self, activity_data: Dict[str, Any], 
                                patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Validate activity levels against clinical recommendations"""
        results = []
        
        # Get patient context
        diagnosis = patient_context.get('diagnosis') if patient_context else None
        recovery_stage = patient_context.get('recovery_stage', 'mid_recovery') if patient_context else 'mid_recovery'
        age = patient_context.get('age', 40) if patient_context else 40
        
        # Steps validation
        if 'steps' in activity_data:
            steps = activity_data['steps']
            
            # Age-based adjustment
            age_factor = self._get_age_adjustment(age)
            
            # Condition-specific validation
            if diagnosis and diagnosis in self.recovery_thresholds:
                stage_thresholds = self.recovery_thresholds[diagnosis][recovery_stage]
                expected_steps = stage_thresholds['steps_per_day']
                
                if steps < expected_steps[0]:
                    results.append({
                        'parameter': 'daily_steps',
                        'value': steps,
                        'severity': ClinicalSeverity.MODERATE,
                        'message': f'Low activity for {diagnosis} {recovery_stage}: {steps} steps',
                        'clinical_significance': 'May indicate pain, fear, or need for encouragement'
                    })
                elif steps > expected_steps[1] * 1.5:
                    results.append({
                        'parameter': 'daily_steps',
                        'value': steps,
                        'severity': ClinicalSeverity.MODERATE,
                        'message': f'High activity for {diagnosis} {recovery_stage}: {steps} steps',
                        'clinical_significance': 'May risk re-injury or overuse'
                    })
        
        return results
    
    def _get_age_adjustment(self, age: int) -> float:
        """Get age-based adjustment factor"""
        for age_range, factor in self.age_adjustments['steps_adjustment'].items():
            if age_range[0] <= age <= age_range[1]:
                return factor
        return 0.6  # Default for very elderly
    
    def validate_recovery_progression(self, current_data: Dict[str, Any], 
                                    historical_data: List[Dict[str, Any]], 
                                    patient_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Validate recovery progression against expected clinical timelines
        
        Args:
            current_data: Current wearable data
            historical_data: Previous data points
            patient_context: Patient context including diagnosis and surgery date
            
        Returns:
            List of validation results regarding recovery progression
        """
        results = []
        
        if not historical_data or not patient_context:
            return results
        
        diagnosis = patient_context.get('diagnosis')
        surgery_date = patient_context.get('surgery_date')
        
        if not diagnosis or not surgery_date:
            return results
        
        # Calculate days since surgery
        days_since_surgery = (datetime.now() - surgery_date).days
        
        # Determine expected recovery stage
        if days_since_surgery <= 30:
            expected_stage = 'early_recovery'
        elif days_since_surgery <= 90:
            expected_stage = 'mid_recovery'
        else:
            expected_stage = 'late_recovery'
        
        # Check if activity progression is appropriate
        if diagnosis in self.recovery_thresholds:
            stage_thresholds = self.recovery_thresholds[diagnosis][expected_stage]
            
            current_steps = current_data.get('steps', 0)
            expected_steps = stage_thresholds['steps_per_day']
            
            if current_steps < expected_steps[0]:
                results.append({
                    'parameter': 'recovery_progression',
                    'value': current_steps,
                    'severity': ClinicalSeverity.MODERATE,
                    'message': f'Activity below expected for {expected_stage} ({days_since_surgery} days post-surgery)',
                    'clinical_significance': 'May indicate complications or need for intervention'
                })
        
        return results
    
    def get_clinical_recommendations(self, validation_results: List[Dict[str, Any]], 
                                   patient_context: Dict[str, Any] = None) -> List[str]:
        """
        Generate clinical recommendations based on validation results
        
        Args:
            validation_results: List of validation results
            patient_context: Patient context
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        # Critical issues require immediate attention
        critical_issues = [r for r in validation_results if r['severity'] == ClinicalSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("URGENT: Contact healthcare provider immediately")
        
        # Severe issues require medical evaluation
        severe_issues = [r for r in validation_results if r['severity'] == ClinicalSeverity.SEVERE]
        if severe_issues:
            recommendations.append("Schedule medical evaluation within 24-48 hours")
        
        # Moderate issues require monitoring
        moderate_issues = [r for r in validation_results if r['severity'] == ClinicalSeverity.MODERATE]
        if moderate_issues:
            recommendations.append("Monitor closely and discuss with healthcare provider at next visit")
        
        # Specific recommendations based on parameter types
        parameter_types = [r['parameter'] for r in validation_results]
        
        if 'heart_rate' in parameter_types:
            recommendations.append("Consider heart rate monitoring and avoid excessive exertion")
        
        if 'sleep' in parameter_types:
            recommendations.append("Focus on sleep hygiene and stress management")
        
        if 'activity' in parameter_types:
            recommendations.append("Adjust activity levels according to recovery stage")
        
        return recommendations