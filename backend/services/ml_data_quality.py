"""
ML Model Data Quality Assurance System
Ensures data meets quality requirements for accurate ML predictions and risk assessments
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from ..models.wearable_data import WearableData
from ..services.data_validation import DataQualityReport, ValidationResult, ValidationSeverity


class MLDataQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class FeatureQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MISSING = "missing"


@dataclass
class FeatureQualityAssessment:
    """Assessment of individual feature quality for ML models"""
    feature_name: str
    quality_score: float
    quality_level: FeatureQuality
    completeness: float
    consistency: float
    reliability: float
    statistical_properties: Dict[str, Any]
    outlier_percentage: float
    recommendation: str
    ml_model_impact: str


@dataclass
class MLDataQualityReport:
    """Comprehensive ML data quality assessment"""
    patient_id: str
    assessment_date: datetime
    overall_ml_quality: MLDataQuality
    overall_score: float
    feature_assessments: List[FeatureQualityAssessment]
    prediction_confidence: float
    model_readiness: Dict[str, bool]
    bias_assessment: Dict[str, Any]
    data_sufficiency: Dict[str, Any]
    recommendations: List[str]
    risk_factors: List[str]
    quality_trends: Dict[str, Any]


class MLDataQualityAssurance:
    """
    Comprehensive ML data quality assurance system
    Ensures data meets requirements for accurate predictions and risk assessments
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ML model requirements
        self.model_requirements = {
            'recovery_velocity': {
                'min_data_points': 14,
                'required_features': ['steps', 'heart_rate', 'sleep_duration'],
                'min_completeness': 0.8,
                'min_consistency': 0.7,
                'temporal_requirement': 'daily'
            },
            'clinical_risk_assessment': {
                'min_data_points': 30,
                'required_features': ['steps', 'heart_rate', 'sleep_duration', 'pain_level'],
                'min_completeness': 0.9,
                'min_consistency': 0.8,
                'temporal_requirement': 'daily'
            },
            'outcome_prediction': {
                'min_data_points': 60,
                'required_features': ['steps', 'heart_rate', 'sleep_duration', 'pain_level', 'rom'],
                'min_completeness': 0.85,
                'min_consistency': 0.75,
                'temporal_requirement': 'daily'
            },
            'anomaly_detection': {
                'min_data_points': 7,
                'required_features': ['steps', 'heart_rate'],
                'min_completeness': 0.7,
                'min_consistency': 0.6,
                'temporal_requirement': 'daily'
            }
        }
        
        # Feature importance weights for different models
        self.feature_importance = {
            'recovery_velocity': {
                'steps': 0.3,
                'heart_rate': 0.2,
                'sleep_duration': 0.2,
                'pain_level': 0.15,
                'rom': 0.15
            },
            'clinical_risk_assessment': {
                'pain_level': 0.25,
                'heart_rate': 0.2,
                'steps': 0.2,
                'sleep_duration': 0.15,
                'rom': 0.1,
                'medication_compliance': 0.1
            },
            'outcome_prediction': {
                'pain_level': 0.2,
                'rom': 0.2,
                'steps': 0.18,
                'heart_rate': 0.15,
                'sleep_duration': 0.12,
                'medication_compliance': 0.15
            }
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6,
            'unusable': 0.0
        }
    
    async def assess_ml_data_quality(self, patient_data: List[WearableData], 
                                   model_type: str = 'all',
                                   patient_context: Dict[str, Any] = None) -> MLDataQualityReport:
        """
        Comprehensive ML data quality assessment
        
        Args:
            patient_data: List of wearable data points
            model_type: Type of ML model to assess for ('all', 'recovery_velocity', etc.)
            patient_context: Patient context for bias assessment
            
        Returns:
            MLDataQualityReport with comprehensive assessment
        """
        
        # Convert to DataFrame for analysis
        df = self._convert_to_dataframe(patient_data)
        
        # Assess individual features
        feature_assessments = await self._assess_feature_quality(df, model_type)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_ml_quality(feature_assessments, model_type)
        overall_quality = self._determine_quality_level(overall_score)
        
        # Assess model readiness
        model_readiness = self._assess_model_readiness(df, feature_assessments)
        
        # Bias assessment
        bias_assessment = await self._assess_bias(df, patient_context)
        
        # Data sufficiency assessment
        data_sufficiency = self._assess_data_sufficiency(df, model_type)
        
        # Calculate prediction confidence
        prediction_confidence = self._calculate_prediction_confidence(feature_assessments, overall_score)
        
        # Generate recommendations
        recommendations = self._generate_ml_recommendations(
            feature_assessments, model_readiness, bias_assessment, data_sufficiency
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(feature_assessments, df)
        
        # Analyze quality trends
        quality_trends = self._analyze_quality_trends(df)
        
        return MLDataQualityReport(
            patient_id=patient_data[0].patient_id if patient_data else "unknown",
            assessment_date=datetime.utcnow(),
            overall_ml_quality=overall_quality,
            overall_score=overall_score,
            feature_assessments=feature_assessments,
            prediction_confidence=prediction_confidence,
            model_readiness=model_readiness,
            bias_assessment=bias_assessment,
            data_sufficiency=data_sufficiency,
            recommendations=recommendations,
            risk_factors=risk_factors,
            quality_trends=quality_trends
        )
    
    def _convert_to_dataframe(self, patient_data: List[WearableData]) -> pd.DataFrame:
        """Convert wearable data to DataFrame for analysis"""
        data_records = []
        
        for data_point in patient_data:
            record = {
                'date': data_point.date,
                'patient_id': data_point.patient_id,
                'steps': data_point.activity_metrics.steps if data_point.activity_metrics else None,
                'distance': data_point.activity_metrics.distance if data_point.activity_metrics else None,
                'calories': data_point.activity_metrics.calories_burned if data_point.activity_metrics else None,
                'heart_rate_avg': data_point.heart_rate_metrics.average_bpm if data_point.heart_rate_metrics else None,
                'heart_rate_max': data_point.heart_rate_metrics.max_bpm if data_point.heart_rate_metrics else None,
                'heart_rate_min': data_point.heart_rate_metrics.min_bpm if data_point.heart_rate_metrics else None,
                'sleep_duration': data_point.sleep_metrics.total_sleep_time if data_point.sleep_metrics else None,
                'deep_sleep': data_point.sleep_metrics.deep_sleep_time if data_point.sleep_metrics else None,
                'rem_sleep': data_point.sleep_metrics.rem_sleep_time if data_point.sleep_metrics else None,
                'walking_speed': data_point.movement_metrics.average_walking_speed if data_point.movement_metrics else None,
                'data_source': data_point.metadata.get('source') if data_point.metadata else None,
                'quality_score': data_point.metadata.get('quality_score', 1.0) if data_point.metadata else 1.0
            }
            data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    async def _assess_feature_quality(self, df: pd.DataFrame, 
                                    model_type: str) -> List[FeatureQualityAssessment]:
        """Assess quality of individual features"""
        assessments = []
        
        # Get relevant features based on model type
        if model_type == 'all':
            features = ['steps', 'heart_rate_avg', 'sleep_duration', 'walking_speed']
        else:
            features = self.model_requirements.get(model_type, {}).get('required_features', [])
        
        for feature in features:
            if feature in df.columns:
                assessment = await self._assess_single_feature(df, feature, model_type)
                assessments.append(assessment)
        
        return assessments
    
    async def _assess_single_feature(self, df: pd.DataFrame, 
                                   feature: str, model_type: str) -> FeatureQualityAssessment:
        """Assess quality of a single feature"""
        
        # Calculate completeness
        completeness = 1.0 - (df[feature].isna().sum() / len(df))
        
        # Calculate consistency (coefficient of variation)
        if completeness > 0:
            clean_data = df[feature].dropna()
            if len(clean_data) > 1 and clean_data.std() > 0:
                consistency = 1.0 - min(clean_data.std() / clean_data.mean(), 1.0)
            else:
                consistency = 1.0
        else:
            consistency = 0.0
        
        # Calculate reliability based on data source
        source_reliability = df.groupby('data_source')['quality_score'].mean()
        reliability = source_reliability.mean() if not source_reliability.empty else 0.8
        
        # Statistical properties
        if completeness > 0:
            clean_data = df[feature].dropna()
            statistical_properties = {
                'mean': clean_data.mean(),
                'std': clean_data.std(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'skewness': stats.skew(clean_data) if len(clean_data) > 2 else 0,
                'kurtosis': stats.kurtosis(clean_data) if len(clean_data) > 2 else 0
            }
        else:
            statistical_properties = {}
        
        # Outlier detection
        outlier_percentage = 0.0
        if completeness > 0.5:
            outlier_percentage = self._detect_outliers(df[feature].dropna())
        
        # Calculate overall quality score
        quality_score = (completeness * 0.4 + consistency * 0.3 + reliability * 0.3) * (1 - outlier_percentage)
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = FeatureQuality.HIGH
        elif quality_score >= 0.7:
            quality_level = FeatureQuality.MEDIUM
        elif quality_score >= 0.5:
            quality_level = FeatureQuality.LOW
        else:
            quality_level = FeatureQuality.MISSING
        
        # Generate recommendation
        recommendation = self._generate_feature_recommendation(
            feature, quality_score, completeness, consistency, outlier_percentage
        )
        
        # ML model impact assessment
        ml_model_impact = self._assess_ml_impact(feature, quality_score, model_type)
        
        return FeatureQualityAssessment(
            feature_name=feature,
            quality_score=quality_score,
            quality_level=quality_level,
            completeness=completeness,
            consistency=consistency,
            reliability=reliability,
            statistical_properties=statistical_properties,
            outlier_percentage=outlier_percentage,
            recommendation=recommendation,
            ml_model_impact=ml_model_impact
        )
    
    def _detect_outliers(self, data: pd.Series) -> float:
        """Detect outliers using IQR method"""
        if len(data) < 4:
            return 0.0
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) / len(data)
    
    def _calculate_overall_ml_quality(self, feature_assessments: List[FeatureQualityAssessment], 
                                    model_type: str) -> float:
        """Calculate overall ML data quality score"""
        if not feature_assessments:
            return 0.0
        
        # Get feature importance weights
        if model_type in self.feature_importance:
            importance_weights = self.feature_importance[model_type]
        else:
            # Equal weights if model type not specified
            importance_weights = {fa.feature_name: 1.0 for fa in feature_assessments}
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for assessment in feature_assessments:
            weight = importance_weights.get(assessment.feature_name, 1.0)
            weighted_score += assessment.quality_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> MLDataQuality:
        """Determine quality level based on score"""
        if score >= self.quality_thresholds['excellent']:
            return MLDataQuality.EXCELLENT
        elif score >= self.quality_thresholds['good']:
            return MLDataQuality.GOOD
        elif score >= self.quality_thresholds['fair']:
            return MLDataQuality.FAIR
        elif score >= self.quality_thresholds['poor']:
            return MLDataQuality.POOR
        else:
            return MLDataQuality.UNUSABLE
    
    def _assess_model_readiness(self, df: pd.DataFrame, 
                              feature_assessments: List[FeatureQualityAssessment]) -> Dict[str, bool]:
        """Assess readiness for different ML models"""
        readiness = {}
        
        for model_name, requirements in self.model_requirements.items():
            # Check minimum data points
            has_sufficient_data = len(df) >= requirements['min_data_points']
            
            # Check required features
            required_features = requirements['required_features']
            feature_quality = {}
            
            for feature in required_features:
                assessment = next((fa for fa in feature_assessments if fa.feature_name == feature), None)
                if assessment:
                    feature_quality[feature] = assessment.quality_score
                else:
                    feature_quality[feature] = 0.0
            
            # Check completeness and consistency requirements
            avg_completeness = np.mean([fa.completeness for fa in feature_assessments])
            avg_consistency = np.mean([fa.consistency for fa in feature_assessments])
            
            meets_completeness = avg_completeness >= requirements['min_completeness']
            meets_consistency = avg_consistency >= requirements['min_consistency']
            
            # Overall readiness
            readiness[model_name] = (
                has_sufficient_data and 
                meets_completeness and 
                meets_consistency and
                all(score >= 0.6 for score in feature_quality.values())
            )
        
        return readiness
    
    async def _assess_bias(self, df: pd.DataFrame, 
                         patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess potential bias in the dataset"""
        bias_assessment = {
            'demographic_bias': {},
            'temporal_bias': {},
            'device_bias': {},
            'overall_bias_risk': 'low'
        }
        
        # Device bias assessment
        if 'data_source' in df.columns:
            device_distribution = df['data_source'].value_counts(normalize=True)
            if len(device_distribution) > 1:
                # Check if one device dominates (>80%)
                max_device_share = device_distribution.max()
                if max_device_share > 0.8:
                    bias_assessment['device_bias'] = {
                        'risk': 'high',
                        'dominant_device': device_distribution.idxmax(),
                        'share': max_device_share,
                        'recommendation': 'Ensure device diversity in training data'
                    }
                else:
                    bias_assessment['device_bias'] = {
                        'risk': 'low',
                        'distribution': device_distribution.to_dict()
                    }
        
        # Temporal bias assessment
        if 'date' in df.columns:
            df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek
            weekday_distribution = df['weekday'].value_counts(normalize=True)
            
            # Check for weekday bias
            if weekday_distribution.std() > 0.1:
                bias_assessment['temporal_bias'] = {
                    'risk': 'medium',
                    'pattern': 'weekday_bias',
                    'recommendation': 'Ensure balanced weekday/weekend representation'
                }
        
        # Determine overall bias risk
        risks = []
        if bias_assessment['device_bias'].get('risk') == 'high':
            risks.append('high')
        if bias_assessment['temporal_bias'].get('risk') == 'medium':
            risks.append('medium')
        
        if 'high' in risks:
            bias_assessment['overall_bias_risk'] = 'high'
        elif 'medium' in risks:
            bias_assessment['overall_bias_risk'] = 'medium'
        else:
            bias_assessment['overall_bias_risk'] = 'low'
        
        return bias_assessment
    
    def _assess_data_sufficiency(self, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Assess data sufficiency for ML models"""
        
        sufficiency = {
            'temporal_coverage': {},
            'feature_coverage': {},
            'data_density': {},
            'overall_sufficiency': 'sufficient'
        }
        
        # Temporal coverage assessment
        if 'date' in df.columns:
            date_range = pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()
            days_covered = date_range.days
            
            # Check for gaps in data
            date_series = pd.to_datetime(df['date']).sort_values()
            gaps = date_series.diff().dt.days
            max_gap = gaps.max() if len(gaps) > 0 else 0
            
            sufficiency['temporal_coverage'] = {
                'days_covered': days_covered,
                'max_gap_days': max_gap,
                'is_sufficient': days_covered >= 30 and max_gap <= 7
            }
        
        # Feature coverage assessment
        feature_coverage = {}
        for col in df.columns:
            if col not in ['date', 'patient_id']:
                completeness = 1.0 - (df[col].isna().sum() / len(df))
                feature_coverage[col] = completeness
        
        sufficiency['feature_coverage'] = feature_coverage
        
        # Data density assessment
        records_per_day = len(df) / max(1, sufficiency['temporal_coverage'].get('days_covered', 1))
        sufficiency['data_density'] = {
            'records_per_day': records_per_day,
            'is_sufficient': records_per_day >= 0.8  # At least 80% daily coverage
        }
        
        # Overall sufficiency
        temporal_sufficient = sufficiency['temporal_coverage'].get('is_sufficient', False)
        density_sufficient = sufficiency['data_density'].get('is_sufficient', False)
        feature_sufficient = np.mean(list(feature_coverage.values())) >= 0.7
        
        if temporal_sufficient and density_sufficient and feature_sufficient:
            sufficiency['overall_sufficiency'] = 'sufficient'
        elif temporal_sufficient or density_sufficient:
            sufficiency['overall_sufficiency'] = 'marginal'
        else:
            sufficiency['overall_sufficiency'] = 'insufficient'
        
        return sufficiency
    
    def _calculate_prediction_confidence(self, feature_assessments: List[FeatureQualityAssessment], 
                                       overall_score: float) -> float:
        """Calculate prediction confidence based on data quality"""
        
        # Base confidence from overall score
        base_confidence = overall_score
        
        # Adjust for feature quality distribution
        quality_scores = [fa.quality_score for fa in feature_assessments]
        if quality_scores:
            quality_std = np.std(quality_scores)
            # Lower confidence if feature quality is inconsistent
            consistency_factor = 1.0 - min(quality_std, 0.3)
            base_confidence *= consistency_factor
        
        # Adjust for completeness
        completeness_scores = [fa.completeness for fa in feature_assessments]
        if completeness_scores:
            avg_completeness = np.mean(completeness_scores)
            base_confidence *= avg_completeness
        
        return min(base_confidence, 1.0)
    
    def _generate_ml_recommendations(self, feature_assessments: List[FeatureQualityAssessment],
                                   model_readiness: Dict[str, bool],
                                   bias_assessment: Dict[str, Any],
                                   data_sufficiency: Dict[str, Any]) -> List[str]:
        """Generate ML-specific recommendations"""
        recommendations = []
        
        # Feature-specific recommendations
        low_quality_features = [fa for fa in feature_assessments if fa.quality_level == FeatureQuality.LOW]
        if low_quality_features:
            recommendations.append(f"Improve data quality for: {', '.join([fa.feature_name for fa in low_quality_features])}")
        
        # Model readiness recommendations
        not_ready_models = [model for model, ready in model_readiness.items() if not ready]
        if not_ready_models:
            recommendations.append(f"Models not ready: {', '.join(not_ready_models)}. Increase data collection period.")
        
        # Bias recommendations
        if bias_assessment['overall_bias_risk'] == 'high':
            recommendations.append("High bias risk detected. Diversify data sources and collection methods.")
        
        # Data sufficiency recommendations
        if data_sufficiency['overall_sufficiency'] == 'insufficient':
            recommendations.append("Insufficient data for reliable ML predictions. Extend data collection period.")
        
        return recommendations
    
    def _identify_risk_factors(self, feature_assessments: List[FeatureQualityAssessment], 
                             df: pd.DataFrame) -> List[str]:
        """Identify risk factors for ML model performance"""
        risk_factors = []
        
        # High outlier percentage
        high_outlier_features = [fa for fa in feature_assessments if fa.outlier_percentage > 0.1]
        if high_outlier_features:
            risk_factors.append(f"High outlier rates in: {', '.join([fa.feature_name for fa in high_outlier_features])}")
        
        # Low consistency
        low_consistency_features = [fa for fa in feature_assessments if fa.consistency < 0.6]
        if low_consistency_features:
            risk_factors.append(f"Low consistency in: {', '.join([fa.feature_name for fa in low_consistency_features])}")
        
        # Sparse data
        if len(df) < 30:
            risk_factors.append("Insufficient historical data for robust predictions")
        
        return risk_factors
    
    def _analyze_quality_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in data quality over time"""
        trends = {
            'quality_improvement': False,
            'data_consistency_trend': 'stable',
            'completeness_trend': 'stable'
        }
        
        if 'date' in df.columns and 'quality_score' in df.columns:
            df_sorted = df.sort_values('date')
            
            # Quality trend analysis
            if len(df_sorted) >= 7:
                recent_quality = df_sorted.tail(7)['quality_score'].mean()
                earlier_quality = df_sorted.head(7)['quality_score'].mean()
                
                if recent_quality > earlier_quality * 1.05:
                    trends['quality_improvement'] = True
                    trends['data_consistency_trend'] = 'improving'
                elif recent_quality < earlier_quality * 0.95:
                    trends['data_consistency_trend'] = 'declining'
        
        return trends
    
    def _generate_feature_recommendation(self, feature: str, quality_score: float, 
                                       completeness: float, consistency: float, 
                                       outlier_percentage: float) -> str:
        """Generate recommendation for a specific feature"""
        
        if quality_score >= 0.9:
            return f"Excellent quality for {feature}. Continue current data collection practices."
        elif quality_score >= 0.7:
            return f"Good quality for {feature}. Minor improvements in consistency recommended."
        elif completeness < 0.7:
            return f"Improve data completeness for {feature}. Current: {completeness:.1%}"
        elif consistency < 0.6:
            return f"Improve data consistency for {feature}. Check device calibration."
        elif outlier_percentage > 0.15:
            return f"High outlier rate for {feature}. Review data collection methodology."
        else:
            return f"Overall improvement needed for {feature}. Focus on completeness and consistency."
    
    def _assess_ml_impact(self, feature: str, quality_score: float, model_type: str) -> str:
        """Assess impact of feature quality on ML model performance"""
        
        # Get feature importance
        if model_type in self.feature_importance:
            importance = self.feature_importance[model_type].get(feature, 0.1)
        else:
            importance = 0.1
        
        impact_score = quality_score * importance
        
        if impact_score >= 0.25:
            return "High positive impact on model accuracy"
        elif impact_score >= 0.15:
            return "Moderate positive impact on model accuracy"
        elif impact_score >= 0.08:
            return "Low impact on model accuracy"
        else:
            return "Minimal impact - may consider excluding from model"