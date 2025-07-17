"""
Clinical Data Validation Engine for Wearable Data
Provides comprehensive validation for healthcare-grade data integrity
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel

from ..models.wearable_data import WearableData, ActivityMetrics, HeartRateMetrics, SleepMetrics, MovementMetrics
from ..utils.clinical_validators import ClinicalValidators


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    PHYSIOLOGICAL = "physiological"
    TEMPORAL = "temporal"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    DEVICE_SPECIFIC = "device_specific"
    CLINICAL = "clinical"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    is_valid: bool
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    field_name: str
    expected_range: Optional[Tuple[float, float]] = None
    actual_value: Optional[Any] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    patient_id: str
    data_id: str
    timestamp: datetime
    overall_score: float
    validation_results: List[ValidationResult]
    completeness_score: float
    consistency_score: float
    reliability_score: float
    clinical_validity_score: float
    recommendations: List[str]
    requires_review: bool = False
    ml_model_ready: bool = True
    research_grade: bool = True


class ClinicalDataValidator:
    """
    Comprehensive clinical-grade data validation engine
    Supports real-time validation with context-aware thresholds
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clinical_validators = ClinicalValidators()
        self.validation_cache = {}
        
        # Clinical thresholds - context-aware and adaptive
        self.physiological_ranges = {
            'heart_rate': {
                'rest': (40, 100),
                'light_activity': (90, 150),
                'moderate_activity': (120, 180),
                'vigorous_activity': (150, 220)
            },
            'steps': {
                'daily_min': 0,
                'daily_max': 50000,
                'hourly_max': 5000
            },
            'sleep': {
                'total_duration': (3, 12),  # hours
                'deep_sleep_ratio': (0.1, 0.3),
                'rem_sleep_ratio': (0.15, 0.25)
            },
            'oxygen_saturation': {
                'normal': (95, 100),
                'concern': (90, 95),
                'critical': (85, 90)
            }
        }
        
        # Device-specific validation rules
        self.device_accuracy_profiles = {
            'HealthKit': {
                'heart_rate_accuracy': 0.95,
                'step_accuracy': 0.98,
                'sleep_accuracy': 0.85
            },
            'Fitbit': {
                'heart_rate_accuracy': 0.92,
                'step_accuracy': 0.96,
                'sleep_accuracy': 0.88
            },
            'manual': {
                'heart_rate_accuracy': 0.70,
                'step_accuracy': 0.80,
                'sleep_accuracy': 0.60
            }
        }
    
    async def validate_wearable_data(self, data: WearableData, 
                                   historical_data: List[WearableData] = None,
                                   patient_context: Dict[str, Any] = None) -> DataQualityReport:
        """
        Comprehensive validation of wearable data with clinical context
        
        Args:
            data: Wearable data to validate
            historical_data: Previous data for temporal validation
            patient_context: Patient-specific context (age, diagnosis, etc.)
            
        Returns:
            DataQualityReport with comprehensive validation results
        """
        validation_results = []
        
        # 1. Physiological Range Validation
        validation_results.extend(await self._validate_physiological_ranges(data, patient_context))
        
        # 2. Cross-metric Consistency Validation
        validation_results.extend(await self._validate_cross_metric_consistency(data))
        
        # 3. Temporal Pattern Validation
        if historical_data:
            validation_results.extend(await self._validate_temporal_patterns(data, historical_data))
        
        # 4. Device-specific Validation
        validation_results.extend(await self._validate_device_specific(data))
        
        # 5. Clinical Context Validation
        if patient_context:
            validation_results.extend(await self._validate_clinical_context(data, patient_context))
        
        # 6. Completeness Validation
        validation_results.extend(await self._validate_completeness(data))
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(validation_results, data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, quality_scores)
        
        # Determine if requires review
        requires_review = any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                            for r in validation_results)
        
        # Check ML model readiness
        ml_model_ready = quality_scores['overall_score'] >= 0.8 and not requires_review
        
        # Check research grade quality
        research_grade = quality_scores['overall_score'] >= 0.9 and quality_scores['completeness_score'] >= 0.95
        
        return DataQualityReport(
            patient_id=data.patient_id,
            data_id=data.id,
            timestamp=datetime.utcnow(),
            overall_score=quality_scores['overall_score'],
            validation_results=validation_results,
            completeness_score=quality_scores['completeness_score'],
            consistency_score=quality_scores['consistency_score'],
            reliability_score=quality_scores['reliability_score'],
            clinical_validity_score=quality_scores['clinical_validity_score'],
            recommendations=recommendations,
            requires_review=requires_review,
            ml_model_ready=ml_model_ready,
            research_grade=research_grade
        )
    
    async def _validate_physiological_ranges(self, data: WearableData, 
                                           patient_context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate physiological parameters within clinical ranges"""
        results = []
        
        # Heart rate validation with activity context
        if data.heart_rate_metrics:
            hr_data = data.heart_rate_metrics
            activity_level = self._determine_activity_level(data.activity_metrics)
            hr_range = self.physiological_ranges['heart_rate'][activity_level]
            
            if hr_data.average_bpm:
                if not hr_range[0] <= hr_data.average_bpm <= hr_range[1]:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PHYSIOLOGICAL,
                        message=f"Heart rate {hr_data.average_bpm} bpm outside expected range for {activity_level}",
                        field_name="heart_rate_metrics.average_bpm",
                        expected_range=hr_range,
                        actual_value=hr_data.average_bpm,
                        confidence_score=0.9
                    ))
        
        # Steps validation
        if data.activity_metrics and data.activity_metrics.steps:
            steps = data.activity_metrics.steps
            max_steps = self.physiological_ranges['steps']['daily_max']
            
            if steps > max_steps:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.PHYSIOLOGICAL,
                    message=f"Daily steps {steps} exceeds physiological maximum",
                    field_name="activity_metrics.steps",
                    expected_range=(0, max_steps),
                    actual_value=steps,
                    confidence_score=0.95
                ))
        
        # Sleep validation
        if data.sleep_metrics:
            sleep_data = data.sleep_metrics
            if sleep_data.total_sleep_time:
                sleep_hours = sleep_data.total_sleep_time / 60  # Convert to hours
                sleep_range = self.physiological_ranges['sleep']['total_duration']
                
                if not sleep_range[0] <= sleep_hours <= sleep_range[1]:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PHYSIOLOGICAL,
                        message=f"Sleep duration {sleep_hours:.1f} hours outside normal range",
                        field_name="sleep_metrics.total_sleep_time",
                        expected_range=sleep_range,
                        actual_value=sleep_hours,
                        confidence_score=0.85
                    ))
        
        return results
    
    async def _validate_cross_metric_consistency(self, data: WearableData) -> List[ValidationResult]:
        """Validate consistency across different metrics"""
        results = []
        
        # Steps vs Distance consistency
        if (data.activity_metrics and 
            data.activity_metrics.steps and 
            data.activity_metrics.distance):
            
            steps = data.activity_metrics.steps
            distance = data.activity_metrics.distance
            
            # Average stride length should be between 0.6-0.8 meters
            stride_length = distance / steps if steps > 0 else 0
            
            if stride_length < 0.4 or stride_length > 1.2:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.CONSISTENCY,
                    message=f"Inconsistent steps ({steps}) and distance ({distance}m) - stride length {stride_length:.2f}m",
                    field_name="activity_metrics",
                    confidence_score=0.8
                ))
        
        # Steps vs Calories consistency
        if (data.activity_metrics and 
            data.activity_metrics.steps and 
            data.activity_metrics.calories_burned):
            
            steps = data.activity_metrics.steps
            calories = data.activity_metrics.calories_burned
            
            # Rough estimation: 0.04-0.06 calories per step
            calories_per_step = calories / steps if steps > 0 else 0
            
            if calories_per_step < 0.02 or calories_per_step > 0.1:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    category=ValidationCategory.CONSISTENCY,
                    message=f"Unusual calories per step ratio: {calories_per_step:.3f}",
                    field_name="activity_metrics",
                    confidence_score=0.7
                ))
        
        return results
    
    async def _validate_temporal_patterns(self, data: WearableData, 
                                        historical_data: List[WearableData]) -> List[ValidationResult]:
        """Validate temporal patterns and realistic progressions"""
        results = []
        
        if not historical_data:
            return results
        
        # Check for realistic daily progression
        recent_data = [d for d in historical_data if d.date >= data.date - timedelta(days=7)]
        
        if recent_data and data.activity_metrics:
            recent_steps = [d.activity_metrics.steps for d in recent_data 
                          if d.activity_metrics and d.activity_metrics.steps]
            
            if recent_steps:
                avg_recent_steps = sum(recent_steps) / len(recent_steps)
                current_steps = data.activity_metrics.steps or 0
                
                # Check for unrealistic jumps (>300% increase)
                if current_steps > avg_recent_steps * 3:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.TEMPORAL,
                        message=f"Unusual step increase: {current_steps} vs recent avg {avg_recent_steps:.0f}",
                        field_name="activity_metrics.steps",
                        confidence_score=0.8
                    ))
        
        return results
    
    async def _validate_device_specific(self, data: WearableData) -> List[ValidationResult]:
        """Validate based on device-specific accuracy profiles"""
        results = []
        
        data_source = data.metadata.get('source', 'unknown') if data.metadata else 'unknown'
        
        if data_source not in self.device_accuracy_profiles:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.DEVICE_SPECIFIC,
                message=f"Unknown data source: {data_source}",
                field_name="metadata.source",
                confidence_score=0.5
            ))
        
        return results
    
    async def _validate_clinical_context(self, data: WearableData, 
                                       patient_context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data within clinical context"""
        results = []
        
        # Age-based validation
        age = patient_context.get('age')
        if age and data.heart_rate_metrics:
            max_hr = 220 - age
            if data.heart_rate_metrics.max_bpm and data.heart_rate_metrics.max_bpm > max_hr:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.CLINICAL,
                    message=f"Max heart rate {data.heart_rate_metrics.max_bpm} exceeds age-predicted maximum {max_hr}",
                    field_name="heart_rate_metrics.max_bpm",
                    confidence_score=0.9
                ))
        
        # Diagnosis-specific validation
        diagnosis = patient_context.get('diagnosis')
        if diagnosis and 'knee' in diagnosis.lower():
            # Knee patients may have limited mobility
            if data.activity_metrics and data.activity_metrics.steps:
                if data.activity_metrics.steps > 15000:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        category=ValidationCategory.CLINICAL,
                        message=f"High activity level for knee diagnosis: {data.activity_metrics.steps} steps",
                        field_name="activity_metrics.steps",
                        confidence_score=0.7
                    ))
        
        return results
    
    async def _validate_completeness(self, data: WearableData) -> List[ValidationResult]:
        """Validate data completeness"""
        results = []
        
        required_fields = ['date', 'patient_id']
        for field in required_fields:
            if not getattr(data, field):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.COMPLETENESS,
                    message=f"Missing required field: {field}",
                    field_name=field,
                    confidence_score=1.0
                ))
        
        # Check for minimal data requirements
        has_meaningful_data = any([
            data.activity_metrics and (data.activity_metrics.steps or data.activity_metrics.distance),
            data.heart_rate_metrics and data.heart_rate_metrics.average_bpm,
            data.sleep_metrics and data.sleep_metrics.total_sleep_time
        ])
        
        if not has_meaningful_data:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.COMPLETENESS,
                message="No meaningful wearable data found",
                field_name="data_completeness",
                confidence_score=0.9
            ))
        
        return results
    
    def _determine_activity_level(self, activity_metrics: Optional[ActivityMetrics]) -> str:
        """Determine activity level based on metrics"""
        if not activity_metrics:
            return 'rest'
        
        steps = activity_metrics.steps or 0
        
        if steps < 2000:
            return 'rest'
        elif steps < 5000:
            return 'light_activity'
        elif steps < 10000:
            return 'moderate_activity'
        else:
            return 'vigorous_activity'
    
    def _calculate_quality_scores(self, validation_results: List[ValidationResult], 
                                data: WearableData) -> Dict[str, float]:
        """Calculate comprehensive quality scores"""
        
        # Overall score based on validation results
        error_weight = {
            ValidationSeverity.INFO: 0.0,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.CRITICAL: 0.5
        }
        
        total_deductions = sum(error_weight.get(r.severity, 0) for r in validation_results)
        overall_score = max(0.0, 1.0 - total_deductions)
        
        # Completeness score
        total_fields = 10  # Approximate number of key fields
        completed_fields = sum([
            1 if data.activity_metrics else 0,
            1 if data.heart_rate_metrics else 0,
            1 if data.sleep_metrics else 0,
            1 if data.movement_metrics else 0,
            1 if data.date else 0,
            1 if data.patient_id else 0
        ])
        completeness_score = completed_fields / total_fields
        
        # Consistency score
        consistency_errors = [r for r in validation_results 
                            if r.category == ValidationCategory.CONSISTENCY and not r.is_valid]
        consistency_score = max(0.0, 1.0 - len(consistency_errors) * 0.2)
        
        # Reliability score based on data source
        data_source = data.metadata.get('source', 'unknown') if data.metadata else 'unknown'
        reliability_score = self.device_accuracy_profiles.get(data_source, {}).get('heart_rate_accuracy', 0.7)
        
        # Clinical validity score
        clinical_errors = [r for r in validation_results 
                         if r.category == ValidationCategory.CLINICAL and not r.is_valid]
        clinical_validity_score = max(0.0, 1.0 - len(clinical_errors) * 0.25)
        
        return {
            'overall_score': overall_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'reliability_score': reliability_score,
            'clinical_validity_score': clinical_validity_score
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                quality_scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical issues
        critical_issues = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("Address critical data quality issues before using for clinical decisions")
        
        # Completeness recommendations
        if quality_scores['completeness_score'] < 0.8:
            recommendations.append("Encourage patient to wear device consistently for better data completeness")
        
        # Consistency recommendations
        if quality_scores['consistency_score'] < 0.7:
            recommendations.append("Review device calibration and patient education on proper usage")
        
        # Clinical validity recommendations
        if quality_scores['clinical_validity_score'] < 0.8:
            recommendations.append("Clinical review recommended for unusual patterns")
        
        return recommendations