"""
Advanced recovery correlation engine for analyzing relationships between 
wearable data metrics and PRO scores in orthopedic recovery.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    metric_pair: Tuple[str, str]
    correlation_coefficient: float
    p_value: float
    sample_size: int
    significance_level: str
    interpretation: str
    confidence_interval: Tuple[float, float]


@dataclass
class TimeSeriesCorrelation:
    """Time-lagged correlation analysis result"""
    wearable_metric: str
    pro_metric: str
    optimal_lag_days: int
    max_correlation: float
    lag_correlations: Dict[int, float]


class RecoveryCorrelationEngine:
    """Engine for analyzing correlations between wearable data and recovery outcomes"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    async def analyze_comprehensive_correlations(self, patient_id: str, days_back: int = 90) -> Dict[str, Any]:
        """
        Comprehensive correlation analysis between wearable metrics and PRO scores
        """
        try:
            # Get patient info to determine PRO type
            patient = await self.db.patients.find_one({"id": patient_id})
            if not patient:
                return {"error": "Patient not found"}
            
            diagnosis_type = patient.get("diagnosis_type")
            is_knee = self._is_knee_patient(diagnosis_type)
            
            # Get synchronized data
            wearable_data, pro_data = await self._get_synchronized_data(patient_id, days_back, is_knee)
            
            if len(wearable_data) < 10 or len(pro_data) < 3:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Perform correlation analyses
            correlations = {
                "pearson_correlations": await self._calculate_pearson_correlations(wearable_data, pro_data),
                "spearman_correlations": await self._calculate_spearman_correlations(wearable_data, pro_data),
                "time_lagged_correlations": await self._calculate_time_lagged_correlations(wearable_data, pro_data),
                "multivariate_correlations": await self._calculate_multivariate_correlations(wearable_data, pro_data),
                "clinical_significance": await self._assess_clinical_significance(wearable_data, pro_data)
            }
            
            # Generate insights
            insights = self._generate_correlation_insights(correlations)
            
            return {
                "patient_id": patient_id,
                "analysis_period_days": days_back,
                "data_points": {
                    "wearable_days": len(wearable_data),
                    "pro_assessments": len(pro_data)
                },
                "correlations": correlations,
                "insights": insights,
                "recommendations": self._generate_correlation_recommendations(correlations),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive correlation analysis failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_activity_pain_correlation(self, patient_id: str) -> Dict[str, Any]:
        """
        Detailed analysis of activity levels vs pain scores
        """
        try:
            # Get data
            wearable_data, pro_data = await self._get_synchronized_data(patient_id, 60)
            
            if not wearable_data or not pro_data:
                return {"error": "Insufficient data"}
            
            # Extract activity and pain metrics
            activity_metrics = self._extract_activity_metrics(wearable_data)
            pain_metrics = self._extract_pain_metrics(pro_data)
            
            # Align data by date
            aligned_data = self._align_data_by_date(activity_metrics, pain_metrics)
            
            if len(aligned_data) < 5:
                return {"error": "Insufficient aligned data points"}
            
            # Calculate correlations
            correlations = {}
            for activity_type in ["steps", "active_minutes", "walking_speed"]:
                if activity_type in aligned_data[0]["activity"]:
                    correlation = self._calculate_correlation_with_significance(
                        [day["activity"][activity_type] for day in aligned_data],
                        [day["pain"] for day in aligned_data]
                    )
                    correlations[f"{activity_type}_pain"] = correlation
            
            # Analyze patterns
            patterns = self._analyze_activity_pain_patterns(aligned_data)
            
            return {
                "patient_id": patient_id,
                "correlations": correlations,
                "patterns": patterns,
                "optimal_activity_range": self._calculate_optimal_activity_range(aligned_data),
                "recommendations": self._generate_activity_recommendations(correlations, patterns)
            }
            
        except Exception as e:
            logger.error(f"Activity-pain correlation analysis failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_sleep_recovery_correlation(self, patient_id: str) -> Dict[str, Any]:
        """
        Analyze correlation between sleep quality and recovery metrics
        """
        try:
            # Get data
            wearable_data, pro_data = await self._get_synchronized_data(patient_id, 60)
            
            # Extract sleep and recovery metrics
            sleep_metrics = self._extract_sleep_metrics(wearable_data)
            recovery_metrics = self._extract_recovery_metrics(pro_data)
            
            # Align data
            aligned_data = self._align_data_by_date(sleep_metrics, recovery_metrics)
            
            if len(aligned_data) < 5:
                return {"error": "Insufficient aligned data points"}
            
            # Calculate correlations
            correlations = {}
            sleep_factors = ["sleep_efficiency", "total_sleep_hours", "deep_sleep_percentage"]
            
            for sleep_factor in sleep_factors:
                if any(sleep_factor in day["sleep"] for day in aligned_data):
                    sleep_values = [day["sleep"].get(sleep_factor, 0) for day in aligned_data]
                    recovery_values = [day["recovery"] for day in aligned_data]
                    
                    correlation = self._calculate_correlation_with_significance(sleep_values, recovery_values)
                    correlations[f"{sleep_factor}_recovery"] = correlation
            
            # Analyze sleep patterns
            sleep_patterns = self._analyze_sleep_patterns(aligned_data)
            
            return {
                "patient_id": patient_id,
                "correlations": correlations,
                "sleep_patterns": sleep_patterns,
                "optimal_sleep_parameters": self._calculate_optimal_sleep_parameters(aligned_data),
                "sleep_recommendations": self._generate_sleep_recommendations(correlations, sleep_patterns)
            }
            
        except Exception as e:
            logger.error(f"Sleep-recovery correlation analysis failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_heart_rate_correlation(self, patient_id: str) -> Dict[str, Any]:
        """
        Analyze heart rate metrics correlation with recovery
        """
        try:
            # Get data
            wearable_data, pro_data = await self._get_synchronized_data(patient_id, 60)
            
            # Extract HR and recovery metrics
            hr_metrics = self._extract_hr_metrics(wearable_data)
            recovery_metrics = self._extract_recovery_metrics(pro_data)
            
            # Align data
            aligned_data = self._align_data_by_date(hr_metrics, recovery_metrics)
            
            if len(aligned_data) < 5:
                return {"error": "Insufficient aligned data points"}
            
            # Calculate correlations
            correlations = {}
            hr_factors = ["resting_hr", "hr_variability", "recovery_hr"]
            
            for hr_factor in hr_factors:
                if any(hr_factor in day["hr"] for day in aligned_data):
                    hr_values = [day["hr"].get(hr_factor, 0) for day in aligned_data]
                    recovery_values = [day["recovery"] for day in aligned_data]
                    
                    # For resting HR, inverse correlation expected (lower is better)
                    if hr_factor == "resting_hr":
                        hr_values = [-v for v in hr_values]
                    
                    correlation = self._calculate_correlation_with_significance(hr_values, recovery_values)
                    correlations[f"{hr_factor}_recovery"] = correlation
            
            # Analyze HR trends
            hr_trends = self._analyze_hr_trends(aligned_data)
            
            return {
                "patient_id": patient_id,
                "correlations": correlations,
                "hr_trends": hr_trends,
                "cardiovascular_insights": self._generate_cardiovascular_insights(correlations, hr_trends),
                "hr_recommendations": self._generate_hr_recommendations(correlations)
            }
            
        except Exception as e:
            logger.error(f"Heart rate correlation analysis failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def calculate_recovery_predictors(self, patient_id: str) -> Dict[str, Any]:
        """
        Identify which wearable metrics are best predictors of recovery outcomes
        """
        try:
            # Get comprehensive data
            wearable_data, pro_data = await self._get_synchronized_data(patient_id, 90)
            
            if len(pro_data) < 5:
                return {"error": "Insufficient PRO data for prediction analysis"}
            
            # Calculate all possible correlations
            all_correlations = []
            
            # Activity predictors
            activity_correlations = await self._calculate_activity_predictors(wearable_data, pro_data)
            all_correlations.extend(activity_correlations)
            
            # Sleep predictors
            sleep_correlations = await self._calculate_sleep_predictors(wearable_data, pro_data)
            all_correlations.extend(sleep_correlations)
            
            # HR predictors
            hr_correlations = await self._calculate_hr_predictors(wearable_data, pro_data)
            all_correlations.extend(hr_correlations)
            
            # Movement predictors
            movement_correlations = await self._calculate_movement_predictors(wearable_data, pro_data)
            all_correlations.extend(movement_correlations)
            
            # Rank predictors by strength and significance
            ranked_predictors = self._rank_predictors(all_correlations)
            
            # Generate prediction model
            prediction_model = self._create_simple_prediction_model(ranked_predictors[:5])
            
            return {
                "patient_id": patient_id,
                "top_predictors": ranked_predictors[:10],
                "prediction_model": prediction_model,
                "model_accuracy": self._calculate_model_accuracy(prediction_model, wearable_data, pro_data),
                "predictor_insights": self._generate_predictor_insights(ranked_predictors),
                "monitoring_recommendations": self._generate_monitoring_recommendations(ranked_predictors)
            }
            
        except Exception as e:
            logger.error(f"Recovery predictor calculation failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    def _is_knee_patient(self, diagnosis_type: str) -> bool:
        """Determine if patient has knee diagnosis"""
        knee_diagnoses = ["ACL Tear", "Meniscus Tear", "Cartilage Defect", "Knee Osteoarthritis", "Post Total Knee Replacement"]
        return diagnosis_type in knee_diagnoses
    
    async def _get_synchronized_data(self, patient_id: str, days_back: int = 90, is_knee: Optional[bool] = None) -> Tuple[List[Dict], List[Dict]]:
        """Get synchronized wearable and PRO data"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Get wearable data
        wearable_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id,
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", 1).to_list(days_back)
        
        # Get PRO data based on patient type
        if is_knee is None:
            patient = await self.db.patients.find_one({"id": patient_id})
            is_knee = self._is_knee_patient(patient.get("diagnosis_type", "")) if patient else True
        
        if is_knee:
            pro_data = await self.db.koos_scores.find({
                "patient_id": patient_id,
                "date": {"$gte": start_date, "$lte": end_date}
            }).sort("date", 1).to_list(100)
        else:
            pro_data = await self.db.ases_scores.find({
                "patient_id": patient_id,
                "date": {"$gte": start_date, "$lte": end_date}
            }).sort("date", 1).to_list(100)
        
        return wearable_data, pro_data
    
    async def _calculate_pearson_correlations(self, wearable_data: List[Dict], pro_data: List[Dict]) -> Dict[str, CorrelationResult]:
        """Calculate Pearson correlations between metrics"""
        correlations = {}
        
        # Align data by date for correlation calculation
        aligned_data = self._align_wearable_pro_data(wearable_data, pro_data)
        
        if len(aligned_data) < 3:
            return correlations
        
        # Define metric pairs to analyze
        wearable_metrics = ["steps", "sleep_efficiency", "walking_speed", "resting_hr"]
        pro_metrics = ["total_score", "pain_score", "function_score"]
        
        for w_metric in wearable_metrics:
            for p_metric in pro_metrics:
                w_values = [self._extract_wearable_metric(day["wearable"], w_metric) for day in aligned_data]
                p_values = [self._extract_pro_metric(day["pro"], p_metric) for day in aligned_data]
                
                # Filter out None values
                valid_pairs = [(w, p) for w, p in zip(w_values, p_values) if w is not None and p is not None]
                
                if len(valid_pairs) >= 3:
                    w_vals, p_vals = zip(*valid_pairs)
                    
                    try:
                        corr_coef, p_value = stats.pearsonr(w_vals, p_vals)
                        
                        # Calculate confidence interval
                        ci = self._calculate_correlation_ci(corr_coef, len(valid_pairs))
                        
                        correlations[f"{w_metric}_{p_metric}"] = CorrelationResult(
                            metric_pair=(w_metric, p_metric),
                            correlation_coefficient=corr_coef,
                            p_value=p_value,
                            sample_size=len(valid_pairs),
                            significance_level=self._get_significance_level(p_value),
                            interpretation=self._interpret_correlation(corr_coef, p_value),
                            confidence_interval=ci
                        )
                    except Exception as e:
                        logger.debug(f"Failed to calculate Pearson correlation for {w_metric}_{p_metric}: {e}")
        
        return correlations
    
    async def _calculate_spearman_correlations(self, wearable_data: List[Dict], pro_data: List[Dict]) -> Dict[str, CorrelationResult]:
        """Calculate Spearman rank correlations (non-parametric)"""
        correlations = {}
        
        # Similar to Pearson but uses Spearman rank correlation
        aligned_data = self._align_wearable_pro_data(wearable_data, pro_data)
        
        if len(aligned_data) < 3:
            return correlations
        
        wearable_metrics = ["steps", "sleep_efficiency", "walking_speed", "resting_hr"]
        pro_metrics = ["total_score", "pain_score", "function_score"]
        
        for w_metric in wearable_metrics:
            for p_metric in pro_metrics:
                w_values = [self._extract_wearable_metric(day["wearable"], w_metric) for day in aligned_data]
                p_values = [self._extract_pro_metric(day["pro"], p_metric) for day in aligned_data]
                
                valid_pairs = [(w, p) for w, p in zip(w_values, p_values) if w is not None and p is not None]
                
                if len(valid_pairs) >= 3:
                    w_vals, p_vals = zip(*valid_pairs)
                    
                    try:
                        corr_coef, p_value = stats.spearmanr(w_vals, p_vals)
                        
                        correlations[f"{w_metric}_{p_metric}_spearman"] = CorrelationResult(
                            metric_pair=(w_metric, p_metric),
                            correlation_coefficient=corr_coef,
                            p_value=p_value,
                            sample_size=len(valid_pairs),
                            significance_level=self._get_significance_level(p_value),
                            interpretation=self._interpret_correlation(corr_coef, p_value),
                            confidence_interval=(0, 0)  # Simplified for Spearman
                        )
                    except Exception as e:
                        logger.debug(f"Failed to calculate Spearman correlation for {w_metric}_{p_metric}: {e}")
        
        return correlations
    
    async def _calculate_time_lagged_correlations(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[TimeSeriesCorrelation]:
        """Calculate time-lagged correlations to find optimal prediction windows"""
        lagged_correlations = []
        
        # Test different lag periods (0 to 14 days)
        max_lag = 14
        
        wearable_metrics = ["steps", "sleep_efficiency", "walking_speed"]
        pro_metrics = ["total_score", "pain_score"]
        
        for w_metric in wearable_metrics:
            for p_metric in pro_metrics:
                lag_results = {}
                
                for lag_days in range(max_lag + 1):
                    correlation = self._calculate_lagged_correlation(
                        wearable_data, pro_data, w_metric, p_metric, lag_days
                    )
                    if correlation is not None:
                        lag_results[lag_days] = correlation
                
                if lag_results:
                    # Find optimal lag
                    optimal_lag = max(lag_results.keys(), key=lambda k: abs(lag_results[k]))
                    max_correlation = lag_results[optimal_lag]
                    
                    lagged_correlations.append(TimeSeriesCorrelation(
                        wearable_metric=w_metric,
                        pro_metric=p_metric,
                        optimal_lag_days=optimal_lag,
                        max_correlation=max_correlation,
                        lag_correlations=lag_results
                    ))
        
        return lagged_correlations
    
    async def _calculate_multivariate_correlations(self, wearable_data: List[Dict], pro_data: List[Dict]) -> Dict[str, Any]:
        """Calculate multivariate correlations between multiple wearable metrics and PRO scores"""
        # Simplified multivariate analysis
        # In practice, this would use more sophisticated methods like canonical correlation
        
        aligned_data = self._align_wearable_pro_data(wearable_data, pro_data)
        
        if len(aligned_data) < 5:
            return {"error": "Insufficient data for multivariate analysis"}
        
        # Create feature matrix
        features = []
        targets = []
        
        for day in aligned_data:
            feature_vector = []
            
            # Extract multiple wearable features
            steps = self._extract_wearable_metric(day["wearable"], "steps")
            sleep_eff = self._extract_wearable_metric(day["wearable"], "sleep_efficiency")
            walking_speed = self._extract_wearable_metric(day["wearable"], "walking_speed")
            
            if all(v is not None for v in [steps, sleep_eff, walking_speed]):
                feature_vector = [steps, sleep_eff, walking_speed]
                target = self._extract_pro_metric(day["pro"], "total_score")
                
                if target is not None:
                    features.append(feature_vector)
                    targets.append(target)
        
        if len(features) < 3:
            return {"error": "Insufficient complete data points"}
        
        # Calculate multiple correlation coefficient
        try:
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Simple multiple correlation calculation
            correlation_matrix = np.corrcoef(features_array.T, targets_array)
            
            return {
                "feature_correlations": {
                    "steps_total_score": float(correlation_matrix[0, -1]),
                    "sleep_efficiency_total_score": float(correlation_matrix[1, -1]),
                    "walking_speed_total_score": float(correlation_matrix[2, -1])
                },
                "sample_size": len(features),
                "feature_intercorrelations": {
                    "steps_sleep": float(correlation_matrix[0, 1]),
                    "steps_walking": float(correlation_matrix[0, 2]),
                    "sleep_walking": float(correlation_matrix[1, 2])
                }
            }
            
        except Exception as e:
            logger.error(f"Multivariate correlation calculation failed: {e}")
            return {"error": str(e)}
    
    async def _assess_clinical_significance(self, wearable_data: List[Dict], pro_data: List[Dict]) -> Dict[str, Any]:
        """Assess clinical significance of correlations"""
        # Define clinically meaningful thresholds
        clinical_thresholds = {
            "steps": {"minimal": 500, "moderate": 1000, "substantial": 2000},
            "sleep_efficiency": {"minimal": 5, "moderate": 10, "substantial": 15},
            "walking_speed": {"minimal": 0.1, "moderate": 0.2, "substantial": 0.3}
        }
        
        pro_thresholds = {
            "total_score": {"minimal": 5, "moderate": 10, "substantial": 15},
            "pain_score": {"minimal": 10, "moderate": 15, "substantial": 20}
        }
        
        # Calculate effect sizes for meaningful changes
        effect_sizes = {}
        
        aligned_data = self._align_wearable_pro_data(wearable_data, pro_data)
        
        for metric, thresholds in clinical_thresholds.items():
            # Find cases where wearable metric improved by substantial amount
            substantial_improvements = []
            corresponding_pro_changes = []
            
            for i in range(1, len(aligned_data)):
                current_w = self._extract_wearable_metric(aligned_data[i]["wearable"], metric)
                previous_w = self._extract_wearable_metric(aligned_data[i-1]["wearable"], metric)
                
                if current_w is not None and previous_w is not None:
                    improvement = current_w - previous_w
                    
                    if improvement >= thresholds["substantial"]:
                        current_pro = self._extract_pro_metric(aligned_data[i]["pro"], "total_score")
                        previous_pro = self._extract_pro_metric(aligned_data[i-1]["pro"], "total_score")
                        
                        if current_pro is not None and previous_pro is not None:
                            pro_change = current_pro - previous_pro
                            substantial_improvements.append(improvement)
                            corresponding_pro_changes.append(pro_change)
            
            if substantial_improvements:
                avg_pro_response = np.mean(corresponding_pro_changes)
                effect_sizes[metric] = {
                    "avg_wearable_improvement": np.mean(substantial_improvements),
                    "avg_pro_response": avg_pro_response,
                    "response_rate": len([p for p in corresponding_pro_changes if p > 0]) / len(corresponding_pro_changes),
                    "clinical_significance": "high" if avg_pro_response > pro_thresholds["total_score"]["moderate"] else "moderate"
                }
        
        return effect_sizes
    
    def _align_wearable_pro_data(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[Dict]:
        """Align wearable and PRO data by date"""
        aligned = []
        
        # Create date lookup for PRO data
        pro_by_date = {item["date"]: item for item in pro_data}
        
        for wearable_day in wearable_data:
            wearable_date = wearable_day["date"]
            
            # Find closest PRO assessment (within 7 days)
            closest_pro = None
            min_days_diff = float('inf')
            
            for pro_date, pro_item in pro_by_date.items():
                if isinstance(pro_date, datetime):
                    pro_date = pro_date.date()
                if isinstance(wearable_date, datetime):
                    wearable_date = wearable_date.date()
                
                days_diff = abs((wearable_date - pro_date).days)
                if days_diff <= 7 and days_diff < min_days_diff:
                    closest_pro = pro_item
                    min_days_diff = days_diff
            
            if closest_pro:
                aligned.append({
                    "date": wearable_date,
                    "wearable": wearable_day,
                    "pro": closest_pro,
                    "days_apart": min_days_diff
                })
        
        return aligned
    
    def _extract_wearable_metric(self, wearable_day: Dict, metric: str) -> Optional[float]:
        """Extract specific metric from wearable data"""
        if metric == "steps":
            activity = wearable_day.get("activity_metrics", {})
            return activity.get("steps") if activity else None
        elif metric == "sleep_efficiency":
            sleep = wearable_day.get("sleep_metrics", {})
            return sleep.get("sleep_efficiency") if sleep else None
        elif metric == "walking_speed":
            movement = wearable_day.get("movement_metrics", {})
            return movement.get("walking_speed_ms") if movement else None
        elif metric == "resting_hr":
            hr = wearable_day.get("heart_rate_metrics", {})
            return hr.get("resting_hr") if hr else None
        
        return None
    
    def _extract_pro_metric(self, pro_day: Dict, metric: str) -> Optional[float]:
        """Extract specific metric from PRO data"""
        if metric == "total_score":
            return pro_day.get("total_score")
        elif metric == "pain_score":
            return pro_day.get("pain_score") or pro_day.get("pain_component")
        elif metric == "function_score":
            return pro_day.get("adl_score") or pro_day.get("function_component")
        
        return None
    
    def _calculate_correlation_ci(self, correlation: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        if n < 3:
            return (correlation, correlation)
        
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se = 1 / np.sqrt(n - 3)
        
        # Critical value for 95% confidence
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (float(r_lower), float(r_upper))
    
    def _get_significance_level(self, p_value: float) -> str:
        """Categorize statistical significance"""
        if p_value < 0.001:
            return "highly_significant"
        elif p_value < 0.01:
            return "very_significant"
        elif p_value < 0.05:
            return "significant"
        elif p_value < 0.10:
            return "marginally_significant"
        else:
            return "not_significant"
    
    def _interpret_correlation(self, correlation: float, p_value: float) -> str:
        """Provide interpretation of correlation strength and significance"""
        if p_value >= 0.05:
            return "No significant correlation"
        
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        return f"{strength.capitalize()} {direction} correlation"
    
    def _calculate_lagged_correlation(self, wearable_data: List[Dict], pro_data: List[Dict], 
                                   w_metric: str, p_metric: str, lag_days: int) -> Optional[float]:
        """Calculate correlation with time lag"""
        try:
            # Shift wearable data by lag_days
            wearable_values = []
            pro_values = []
            
            for i, pro_day in enumerate(pro_data):
                pro_date = pro_day["date"]
                if isinstance(pro_date, datetime):
                    pro_date = pro_date.date()
                
                # Find wearable data lag_days before this PRO assessment
                target_date = pro_date - timedelta(days=lag_days)
                
                # Find closest wearable data to target date
                closest_wearable = None
                min_diff = float('inf')
                
                for w_day in wearable_data:
                    w_date = w_day["date"]
                    if isinstance(w_date, datetime):
                        w_date = w_date.date()
                    
                    diff = abs((w_date - target_date).days)
                    if diff <= 2 and diff < min_diff:  # Within 2 days
                        closest_wearable = w_day
                        min_diff = diff
                
                if closest_wearable:
                    w_value = self._extract_wearable_metric(closest_wearable, w_metric)
                    p_value = self._extract_pro_metric(pro_day, p_metric)
                    
                    if w_value is not None and p_value is not None:
                        wearable_values.append(w_value)
                        pro_values.append(p_value)
            
            if len(wearable_values) >= 3:
                correlation, _ = stats.pearsonr(wearable_values, pro_values)
                return correlation
            
        except Exception as e:
            logger.debug(f"Failed to calculate lagged correlation: {e}")
        
        return None
    
    def _generate_correlation_insights(self, correlations: Dict) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        # Analyze Pearson correlations
        pearson_corrs = correlations.get("pearson_correlations", {})
        
        # Find strongest correlations
        significant_correlations = [
            (key, corr) for key, corr in pearson_corrs.items()
            if corr.p_value < 0.05 and abs(corr.correlation_coefficient) > 0.3
        ]
        
        if significant_correlations:
            strongest = max(significant_correlations, key=lambda x: abs(x[1].correlation_coefficient))
            insights.append(f"Strongest correlation: {strongest[0]} (r={strongest[1].correlation_coefficient:.3f})")
        
        # Analyze time-lagged correlations
        lagged_corrs = correlations.get("time_lagged_correlations", [])
        for lagged in lagged_corrs:
            if abs(lagged.max_correlation) > 0.4:
                insights.append(
                    f"{lagged.wearable_metric} shows optimal correlation with {lagged.pro_metric} "
                    f"at {lagged.optimal_lag_days} day lag (r={lagged.max_correlation:.3f})"
                )
        
        # Clinical significance insights
        clinical_sig = correlations.get("clinical_significance", {})
        for metric, significance in clinical_sig.items():
            if significance.get("clinical_significance") == "high":
                insights.append(f"{metric} improvements show clinically meaningful PRO responses")
        
        return insights
    
    def _generate_correlation_recommendations(self, correlations: Dict) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        
        # Check for strong activity correlations
        pearson_corrs = correlations.get("pearson_correlations", {})
        
        for key, corr in pearson_corrs.items():
            if corr.p_value < 0.05 and corr.correlation_coefficient > 0.4:
                if "steps" in key:
                    recommendations.append("Focus on increasing daily step count for improved outcomes")
                elif "sleep_efficiency" in key:
                    recommendations.append("Prioritize sleep quality optimization")
                elif "walking_speed" in key:
                    recommendations.append("Work on improving walking speed and gait quality")
        
        # Time-lagged recommendations
        lagged_corrs = correlations.get("time_lagged_correlations", [])
        for lagged in lagged_corrs:
            if abs(lagged.max_correlation) > 0.4 and lagged.optimal_lag_days > 0:
                recommendations.append(
                    f"Monitor {lagged.wearable_metric} as early indicator "
                    f"({lagged.optimal_lag_days} days ahead) of {lagged.pro_metric} changes"
                )
        
        if not recommendations:
            recommendations.append("Continue monitoring all metrics for pattern identification")
        
        return recommendations
    
    # Additional helper methods for specific correlation analyses
    def _extract_activity_metrics(self, wearable_data: List[Dict]) -> List[Dict]:
        """Extract activity metrics with dates"""
        activity_data = []
        for day in wearable_data:
            activity = day.get("activity_metrics", {})
            if activity:
                activity_data.append({
                    "date": day["date"],
                    "steps": activity.get("steps"),
                    "active_minutes": activity.get("active_minutes"),
                    "walking_speed": day.get("movement_metrics", {}).get("walking_speed_ms")
                })
        return activity_data
    
    def _extract_pain_metrics(self, pro_data: List[Dict]) -> List[Dict]:
        """Extract pain metrics with dates"""
        pain_data = []
        for day in pro_data:
            pain_score = day.get("pain_score") or day.get("pain_component")
            if pain_score is not None:
                pain_data.append({
                    "date": day["date"],
                    "pain": pain_score
                })
        return pain_data
    
    def _extract_sleep_metrics(self, wearable_data: List[Dict]) -> List[Dict]:
        """Extract sleep metrics with dates"""
        sleep_data = []
        for day in wearable_data:
            sleep = day.get("sleep_metrics", {})
            if sleep:
                sleep_data.append({
                    "date": day["date"],
                    "sleep_efficiency": sleep.get("sleep_efficiency"),
                    "total_sleep_hours": sleep.get("total_sleep_minutes", 0) / 60 if sleep.get("total_sleep_minutes") else None,
                    "deep_sleep_percentage": (sleep.get("deep_sleep_minutes", 0) / sleep.get("total_sleep_minutes", 1)) * 100 if sleep.get("total_sleep_minutes") else None
                })
        return sleep_data
    
    def _extract_recovery_metrics(self, pro_data: List[Dict]) -> List[Dict]:
        """Extract recovery metrics with dates"""
        recovery_data = []
        for day in pro_data:
            total_score = day.get("total_score")
            if total_score is not None:
                recovery_data.append({
                    "date": day["date"],
                    "recovery": total_score
                })
        return recovery_data
    
    def _extract_hr_metrics(self, wearable_data: List[Dict]) -> List[Dict]:
        """Extract heart rate metrics with dates"""
        hr_data = []
        for day in wearable_data:
            hr = day.get("heart_rate_metrics", {})
            if hr:
                hr_data.append({
                    "date": day["date"],
                    "resting_hr": hr.get("resting_hr"),
                    "hr_variability": hr.get("hr_variability_ms"),
                    "recovery_hr": hr.get("recovery_hr")
                })
        return hr_data
    
    def _align_data_by_date(self, primary_data: List[Dict], secondary_data: List[Dict]) -> List[Dict]:
        """Align two datasets by date"""
        aligned = []
        
        # Create lookup for secondary data
        secondary_by_date = {item["date"]: item for item in secondary_data}
        
        for primary_item in primary_data:
            primary_date = primary_item["date"]
            
            # Find exact or closest match
            if primary_date in secondary_by_date:
                aligned.append({
                    "date": primary_date,
                    primary_data[0].__class__.__name__.split("_")[0].lower(): {k: v for k, v in primary_item.items() if k != "date"},
                    secondary_data[0].__class__.__name__.split("_")[0].lower(): {k: v for k, v in secondary_by_date[primary_date].items() if k != "date"}
                })
        
        return aligned
    
    def _calculate_correlation_with_significance(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """Calculate correlation with significance testing"""
        try:
            # Remove None values
            valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
            
            if len(valid_pairs) < 3:
                return {"correlation": 0, "p_value": 1, "significance": "insufficient_data"}
            
            x_vals, y_vals = zip(*valid_pairs)
            correlation, p_value = stats.pearsonr(x_vals, y_vals)
            
            return {
                "correlation": correlation,
                "p_value": p_value,
                "significance": self._get_significance_level(p_value),
                "sample_size": len(valid_pairs)
            }
            
        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return {"correlation": 0, "p_value": 1, "significance": "error"}
    
    # Additional placeholder methods for comprehensive analysis
    async def _calculate_activity_predictors(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[Dict]:
        """Calculate activity-based predictors"""
        # Placeholder implementation
        return []
    
    async def _calculate_sleep_predictors(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[Dict]:
        """Calculate sleep-based predictors"""
        # Placeholder implementation
        return []
    
    async def _calculate_hr_predictors(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[Dict]:
        """Calculate heart rate-based predictors"""
        # Placeholder implementation
        return []
    
    async def _calculate_movement_predictors(self, wearable_data: List[Dict], pro_data: List[Dict]) -> List[Dict]:
        """Calculate movement-based predictors"""
        # Placeholder implementation
        return []
    
    def _rank_predictors(self, all_correlations: List[Dict]) -> List[Dict]:
        """Rank predictors by strength and significance"""
        # Placeholder implementation
        return []
    
    def _create_simple_prediction_model(self, top_predictors: List[Dict]) -> Dict[str, Any]:
        """Create simple prediction model"""
        # Placeholder implementation
        return {"model_type": "linear", "predictors": top_predictors}
    
    def _calculate_model_accuracy(self, model: Dict, wearable_data: List[Dict], pro_data: List[Dict]) -> float:
        """Calculate model accuracy"""
        # Placeholder implementation
        return 0.75
    
    def _generate_predictor_insights(self, predictors: List[Dict]) -> List[str]:
        """Generate insights about predictors"""
        # Placeholder implementation
        return ["Activity level is a key predictor of recovery outcomes"]
    
    def _generate_monitoring_recommendations(self, predictors: List[Dict]) -> List[str]:
        """Generate monitoring recommendations"""
        # Placeholder implementation
        return ["Focus on consistent activity tracking"]
    
    def _analyze_activity_pain_patterns(self, aligned_data: List[Dict]) -> Dict[str, Any]:
        """Analyze activity-pain patterns"""
        # Placeholder implementation
        return {"pattern_type": "inverse_correlation"}
    
    def _calculate_optimal_activity_range(self, aligned_data: List[Dict]) -> Dict[str, Any]:
        """Calculate optimal activity range"""
        # Placeholder implementation
        return {"min_steps": 3000, "max_steps": 8000}
    
    def _generate_activity_recommendations(self, correlations: Dict, patterns: Dict) -> List[str]:
        """Generate activity recommendations"""
        # Placeholder implementation
        return ["Maintain consistent daily activity within optimal range"]
    
    def _analyze_sleep_patterns(self, aligned_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sleep patterns"""
        # Placeholder implementation
        return {"sleep_quality_trend": "improving"}
    
    def _calculate_optimal_sleep_parameters(self, aligned_data: List[Dict]) -> Dict[str, Any]:
        """Calculate optimal sleep parameters"""
        # Placeholder implementation
        return {"optimal_efficiency": 85, "optimal_duration": 8}
    
    def _generate_sleep_recommendations(self, correlations: Dict, patterns: Dict) -> List[str]:
        """Generate sleep recommendations"""
        # Placeholder implementation
        return ["Aim for 85% sleep efficiency or higher"]
    
    def _analyze_hr_trends(self, aligned_data: List[Dict]) -> Dict[str, Any]:
        """Analyze heart rate trends"""
        # Placeholder implementation
        return {"resting_hr_trend": "decreasing"}
    
    def _generate_cardiovascular_insights(self, correlations: Dict, trends: Dict) -> List[str]:
        """Generate cardiovascular insights"""
        # Placeholder implementation
        return ["Cardiovascular fitness is improving"]
    
    def _generate_hr_recommendations(self, correlations: Dict) -> List[str]:
        """Generate heart rate recommendations"""
        # Placeholder implementation
        return ["Monitor resting heart rate trends"]