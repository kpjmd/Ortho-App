"""
Recovery-specific metrics and calculations for orthopedic rehabilitation.
Provides specialized metric calculations and analysis functions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from enum import Enum
import math
from dataclasses import dataclass


class RecoveryPhase(str, Enum):
    """Recovery phases for orthopedic rehabilitation"""
    ACUTE = "acute"  # 0-2 weeks post-surgery
    EARLY = "early"  # 2-6 weeks post-surgery  
    INTERMEDIATE = "intermediate"  # 6-12 weeks post-surgery
    LATE = "late"  # 12-26 weeks post-surgery
    MAINTENANCE = "maintenance"  # 26+ weeks post-surgery


class MetricCategory(str, Enum):
    """Categories of recovery metrics"""
    ACTIVITY = "activity"
    SLEEP = "sleep"
    PAIN = "pain"
    FUNCTION = "function"
    MOBILITY = "mobility"
    CARDIOVASCULAR = "cardiovascular"


@dataclass
class MetricTrend:
    """Represents a trend in a recovery metric"""
    metric_name: str
    trend_direction: str  # "improving", "stable", "declining"
    trend_magnitude: float  # How strong the trend is
    trend_significance: float  # Statistical significance
    trend_velocity: float  # Rate of change per day/week
    confidence_level: str  # "high", "medium", "low"


@dataclass
class RecoveryMilestone:
    """Represents a recovery milestone"""
    milestone_id: str
    name: str
    description: str
    target_week: int
    metric_type: MetricCategory
    target_value: float
    tolerance_range: Tuple[float, float]
    critical: bool
    achieved: bool
    achievement_date: Optional[date]


class RecoveryMetricsCalculator:
    """Calculator for recovery-specific metrics and analysis"""
    
    @staticmethod
    def calculate_recovery_velocity(
        current_values: List[float], 
        previous_values: List[float], 
        time_interval_days: int = 7
    ) -> Dict[str, float]:
        """
        Calculate recovery velocity (rate of improvement)
        """
        if len(current_values) != len(previous_values) or not current_values:
            return {"velocity": 0.0, "acceleration": 0.0, "confidence": 0.0}
        
        # Calculate mean change
        current_mean = np.mean(current_values)
        previous_mean = np.mean(previous_values)
        
        # Velocity = change per day
        velocity = (current_mean - previous_mean) / time_interval_days
        
        # Calculate acceleration (change in velocity)
        if len(current_values) >= 3 and len(previous_values) >= 3:
            current_trend = np.polyfit(range(len(current_values)), current_values, 1)[0]
            previous_trend = np.polyfit(range(len(previous_values)), previous_values, 1)[0]
            acceleration = (current_trend - previous_trend) / time_interval_days
        else:
            acceleration = 0.0
        
        # Confidence based on variance
        current_var = np.var(current_values) if len(current_values) > 1 else 0
        previous_var = np.var(previous_values) if len(previous_values) > 1 else 0
        avg_var = (current_var + previous_var) / 2
        
        # Lower variance = higher confidence
        confidence = max(0, 1 - (avg_var / (current_mean + 1))) if current_mean > 0 else 0
        
        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "confidence": confidence,
            "current_mean": current_mean,
            "previous_mean": previous_mean
        }
    
    @staticmethod
    def calculate_activity_consistency_score(step_counts: List[int], days: int = 7) -> Dict[str, float]:
        """
        Calculate activity consistency score (0-100)
        Higher scores indicate more consistent activity patterns
        """
        if len(step_counts) < days:
            return {"consistency_score": 0.0, "reason": "insufficient_data"}
        
        recent_steps = step_counts[-days:]
        
        # Calculate coefficient of variation (CV)
        mean_steps = np.mean(recent_steps)
        std_steps = np.std(recent_steps)
        
        if mean_steps == 0:
            return {"consistency_score": 0.0, "reason": "no_activity"}
        
        cv = std_steps / mean_steps
        
        # Convert CV to consistency score (lower CV = higher consistency)
        # CV of 0.3 or less = excellent consistency (score 90-100)
        # CV of 0.5 or less = good consistency (score 70-90)
        # CV of 0.8 or less = fair consistency (score 50-70)
        # CV > 0.8 = poor consistency (score 0-50)
        
        if cv <= 0.3:
            consistency_score = 90 + (0.3 - cv) * 33.33  # Scale to 90-100
        elif cv <= 0.5:
            consistency_score = 70 + (0.5 - cv) * 100    # Scale to 70-90
        elif cv <= 0.8:
            consistency_score = 50 + (0.8 - cv) * 66.67  # Scale to 50-70
        else:
            consistency_score = max(0, 50 - (cv - 0.8) * 62.5)  # Scale to 0-50
        
        return {
            "consistency_score": min(100, max(0, consistency_score)),
            "coefficient_of_variation": cv,
            "mean_daily_steps": mean_steps,
            "std_daily_steps": std_steps,
            "assessment_days": days
        }
    
    @staticmethod
    def calculate_sleep_quality_index(
        sleep_efficiency: List[float],
        sleep_duration: List[float],
        wake_episodes: List[int]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive sleep quality index (0-100)
        """
        if not all([sleep_efficiency, sleep_duration, wake_episodes]):
            return {"sleep_quality_index": 0.0, "reason": "insufficient_data"}
        
        min_length = min(len(sleep_efficiency), len(sleep_duration), len(wake_episodes))
        
        # Truncate to same length
        sleep_eff = sleep_efficiency[:min_length]
        sleep_dur = sleep_duration[:min_length]
        wake_eps = wake_episodes[:min_length]
        
        # Component scores (0-100 each)
        
        # Efficiency score (target: 85%+)
        avg_efficiency = np.mean(sleep_eff)
        efficiency_score = min(100, (avg_efficiency / 85) * 100) if avg_efficiency > 0 else 0
        
        # Duration score (target: 7-9 hours)
        avg_duration = np.mean(sleep_dur)
        if 7 <= avg_duration <= 9:
            duration_score = 100
        elif 6 <= avg_duration < 7 or 9 < avg_duration <= 10:
            duration_score = 80
        elif 5 <= avg_duration < 6 or 10 < avg_duration <= 11:
            duration_score = 60
        else:
            duration_score = max(0, 40 - abs(avg_duration - 8) * 10)
        
        # Wake episodes score (target: <3 per night)
        avg_wake_episodes = np.mean(wake_eps)
        if avg_wake_episodes <= 2:
            wake_score = 100
        elif avg_wake_episodes <= 4:
            wake_score = 80
        elif avg_wake_episodes <= 6:
            wake_score = 60
        else:
            wake_score = max(0, 60 - (avg_wake_episodes - 6) * 10)
        
        # Consistency score
        eff_consistency = 1 - (np.std(sleep_eff) / np.mean(sleep_eff)) if np.mean(sleep_eff) > 0 else 0
        dur_consistency = 1 - (np.std(sleep_dur) / np.mean(sleep_dur)) if np.mean(sleep_dur) > 0 else 0
        consistency_score = ((eff_consistency + dur_consistency) / 2) * 100
        
        # Weighted composite score
        composite_score = (
            efficiency_score * 0.35 +
            duration_score * 0.25 +
            wake_score * 0.25 +
            consistency_score * 0.15
        )
        
        return {
            "sleep_quality_index": composite_score,
            "efficiency_score": efficiency_score,
            "duration_score": duration_score,
            "wake_episodes_score": wake_score,
            "consistency_score": consistency_score,
            "avg_efficiency": avg_efficiency,
            "avg_duration": avg_duration,
            "avg_wake_episodes": avg_wake_episodes
        }
    
    @staticmethod
    def calculate_pain_function_ratio(pain_scores: List[float], function_scores: List[float]) -> Dict[str, float]:
        """
        Calculate pain-to-function ratio to assess recovery balance
        """
        if not pain_scores or not function_scores:
            return {"ratio": 0.0, "interpretation": "insufficient_data"}
        
        min_length = min(len(pain_scores), len(function_scores))
        pain = pain_scores[:min_length]
        function = function_scores[:min_length]
        
        # Convert pain scores to "pain level" (invert if needed)
        # Assuming higher pain scores = less pain (KOOS/ASES style)
        pain_level = [100 - p for p in pain]  # Convert to pain intensity
        
        avg_pain_level = np.mean(pain_level)
        avg_function = np.mean(function)
        
        if avg_function == 0:
            return {"ratio": float('inf'), "interpretation": "no_function"}
        
        # Ratio of pain intensity to function level
        ratio = avg_pain_level / avg_function
        
        # Interpretation
        if ratio < 0.3:
            interpretation = "excellent_balance"  # Low pain, good function
        elif ratio < 0.5:
            interpretation = "good_balance"
        elif ratio < 0.8:
            interpretation = "fair_balance"
        elif ratio < 1.2:
            interpretation = "poor_balance"
        else:
            interpretation = "very_poor_balance"  # High pain, low function
        
        return {
            "ratio": ratio,
            "interpretation": interpretation,
            "avg_pain_level": avg_pain_level,
            "avg_function_level": avg_function,
            "balance_score": max(0, 100 - (ratio * 50))  # Convert to 0-100 scale
        }
    
    @staticmethod
    def calculate_mobility_progression_index(
        walking_speeds: List[float],
        step_counts: List[int],
        weeks_post_surgery: int,
        diagnosis_type: str
    ) -> Dict[str, float]:
        """
        Calculate mobility progression index based on diagnosis-specific expectations
        """
        if not walking_speeds or not step_counts:
            return {"mobility_index": 0.0, "reason": "insufficient_data"}
        
        min_length = min(len(walking_speeds), len(step_counts))
        speeds = walking_speeds[:min_length]
        steps = step_counts[:min_length]
        
        # Remove None/zero values
        valid_speeds = [s for s in speeds if s is not None and s > 0]
        valid_steps = [s for s in steps if s is not None and s > 0]
        
        if not valid_speeds or not valid_steps:
            return {"mobility_index": 0.0, "reason": "no_valid_data"}
        
        avg_speed = np.mean(valid_speeds)
        avg_steps = np.mean(valid_steps)
        
        # Diagnosis-specific targets
        target_speeds = {
            "ACL Tear": 1.2,  # m/s
            "Meniscus Tear": 1.3,
            "Knee Osteoarthritis": 1.1,
            "Post Total Knee Replacement": 1.0,
            "Rotator Cuff Tear": 1.3,  # Shoulder shouldn't affect walking much
            "Labral Tear": 1.3,
            "Shoulder Instability": 1.3,
            "Post Total Shoulder Replacement": 1.2
        }
        
        target_steps = {
            "ACL Tear": 6000,
            "Meniscus Tear": 7000,
            "Knee Osteoarthritis": 5000,
            "Post Total Knee Replacement": 4000,
            "Rotator Cuff Tear": 7000,
            "Labral Tear": 7000,
            "Shoulder Instability": 7000,
            "Post Total Shoulder Replacement": 6000
        }
        
        target_speed = target_speeds.get(diagnosis_type, 1.2)
        target_step_count = target_steps.get(diagnosis_type, 6000)
        
        # Adjust targets based on recovery phase
        phase_multipliers = {
            "acute": 0.3,      # 0-2 weeks
            "early": 0.5,      # 2-6 weeks
            "intermediate": 0.7, # 6-12 weeks
            "late": 0.9,       # 12-26 weeks
            "maintenance": 1.0  # 26+ weeks
        }
        
        phase = RecoveryMetricsCalculator.determine_recovery_phase(weeks_post_surgery)
        multiplier = phase_multipliers.get(phase.value, 1.0)
        
        adjusted_speed_target = target_speed * multiplier
        adjusted_steps_target = target_step_count * multiplier
        
        # Calculate component scores
        speed_score = min(100, (avg_speed / adjusted_speed_target) * 100)
        steps_score = min(100, (avg_steps / adjusted_steps_target) * 100)
        
        # Composite mobility index
        mobility_index = (speed_score * 0.6 + steps_score * 0.4)
        
        return {
            "mobility_index": mobility_index,
            "speed_score": speed_score,
            "steps_score": steps_score,
            "avg_walking_speed": avg_speed,
            "avg_daily_steps": avg_steps,
            "target_speed": adjusted_speed_target,
            "target_steps": adjusted_steps_target,
            "recovery_phase": phase.value,
            "weeks_post_surgery": weeks_post_surgery
        }
    
    @staticmethod
    def calculate_cardiovascular_recovery_index(
        resting_hrs: List[int],
        hr_variabilities: List[float],
        recovery_hrs: List[int]
    ) -> Dict[str, float]:
        """
        Calculate cardiovascular recovery index
        """
        if not resting_hrs:
            return {"cv_recovery_index": 0.0, "reason": "no_hr_data"}
        
        # Resting HR component (lower is better)
        avg_resting_hr = np.mean(resting_hrs)
        
        # Age-adjusted target (assuming adult population)
        target_resting_hr = 65  # Good fitness level
        resting_hr_score = max(0, 100 - (avg_resting_hr - target_resting_hr) * 2)
        
        # HRV component (higher is better)
        hrv_score = 0
        if hr_variabilities:
            avg_hrv = np.mean([hrv for hrv in hr_variabilities if hrv is not None])
            target_hrv = 35  # Milliseconds
            hrv_score = min(100, (avg_hrv / target_hrv) * 100)
        
        # Recovery HR component (faster recovery is better)
        recovery_hr_score = 0
        if recovery_hrs:
            # This would need more sophisticated calculation
            # For now, assume lower recovery HR indicates better fitness
            avg_recovery_hr = np.mean(recovery_hrs)
            target_recovery_hr = 100
            recovery_hr_score = max(0, 100 - (avg_recovery_hr - target_recovery_hr) * 1.5)
        
        # Trend analysis
        if len(resting_hrs) >= 7:
            # Calculate trend over last week
            recent_trend = np.polyfit(range(len(resting_hrs[-7:])), resting_hrs[-7:], 1)[0]
            trend_score = 50 + (-recent_trend * 10)  # Negative trend is good for resting HR
            trend_score = max(0, min(100, trend_score))
        else:
            trend_score = 50  # Neutral
        
        # Weighted composite
        weights = [0.4, 0.3, 0.2, 0.1]  # Resting HR, HRV, Recovery HR, Trend
        scores = [resting_hr_score, hrv_score, recovery_hr_score, trend_score]
        
        cv_recovery_index = sum(w * s for w, s in zip(weights, scores))
        
        return {
            "cv_recovery_index": cv_recovery_index,
            "resting_hr_score": resting_hr_score,
            "hrv_score": hrv_score,
            "recovery_hr_score": recovery_hr_score,
            "trend_score": trend_score,
            "avg_resting_hr": avg_resting_hr,
            "avg_hrv": np.mean(hr_variabilities) if hr_variabilities else None,
            "avg_recovery_hr": np.mean(recovery_hrs) if recovery_hrs else None
        }
    
    @staticmethod
    def determine_recovery_phase(weeks_post_surgery: int) -> RecoveryPhase:
        """Determine recovery phase based on weeks post-surgery"""
        if weeks_post_surgery <= 2:
            return RecoveryPhase.ACUTE
        elif weeks_post_surgery <= 6:
            return RecoveryPhase.EARLY
        elif weeks_post_surgery <= 12:
            return RecoveryPhase.INTERMEDIATE
        elif weeks_post_surgery <= 26:
            return RecoveryPhase.LATE
        else:
            return RecoveryPhase.MAINTENANCE
    
    @staticmethod
    def calculate_adherence_score(
        expected_sessions: int,
        completed_sessions: int,
        data_collection_days: int,
        expected_data_days: int
    ) -> Dict[str, float]:
        """
        Calculate patient adherence score
        """
        # Exercise adherence
        exercise_adherence = (completed_sessions / expected_sessions * 100) if expected_sessions > 0 else 0
        
        # Data collection adherence
        data_adherence = (data_collection_days / expected_data_days * 100) if expected_data_days > 0 else 0
        
        # Composite adherence score
        adherence_score = (exercise_adherence * 0.6 + data_adherence * 0.4)
        
        # Categorize adherence
        if adherence_score >= 90:
            adherence_category = "excellent"
        elif adherence_score >= 80:
            adherence_category = "good"
        elif adherence_score >= 70:
            adherence_category = "fair"
        elif adherence_score >= 60:
            adherence_category = "poor"
        else:
            adherence_category = "very_poor"
        
        return {
            "adherence_score": adherence_score,
            "adherence_category": adherence_category,
            "exercise_adherence": exercise_adherence,
            "data_adherence": data_adherence,
            "completed_sessions": completed_sessions,
            "expected_sessions": expected_sessions,
            "data_collection_days": data_collection_days,
            "expected_data_days": expected_data_days
        }
    
    @staticmethod
    def calculate_risk_stratification_score(
        age: int,
        bmi: float,
        comorbidities: List[str],
        surgery_type: str,
        baseline_function: float
    ) -> Dict[str, Any]:
        """
        Calculate risk stratification score for recovery outcomes
        """
        risk_score = 0
        risk_factors = []
        
        # Age factor
        if age >= 65:
            risk_score += 15
            risk_factors.append("Advanced age (≥65)")
        elif age >= 55:
            risk_score += 8
            risk_factors.append("Older age (55-64)")
        
        # BMI factor
        if bmi >= 35:
            risk_score += 20
            risk_factors.append("Severe obesity (BMI ≥35)")
        elif bmi >= 30:
            risk_score += 12
            risk_factors.append("Obesity (BMI 30-35)")
        elif bmi >= 25:
            risk_score += 5
            risk_factors.append("Overweight (BMI 25-30)")
        
        # Comorbidities
        high_risk_conditions = ["diabetes", "cardiovascular_disease", "chronic_pain", "depression"]
        moderate_risk_conditions = ["hypertension", "osteoporosis", "arthritis"]
        
        for condition in comorbidities:
            if condition.lower() in high_risk_conditions:
                risk_score += 15
                risk_factors.append(f"High-risk comorbidity: {condition}")
            elif condition.lower() in moderate_risk_conditions:
                risk_score += 8
                risk_factors.append(f"Moderate-risk comorbidity: {condition}")
        
        # Surgery complexity
        complex_surgeries = ["Total Knee Replacement", "Total Shoulder Replacement", "ACL Reconstruction"]
        if surgery_type in complex_surgeries:
            risk_score += 10
            risk_factors.append("Complex surgical procedure")
        
        # Baseline function
        if baseline_function < 40:
            risk_score += 15
            risk_factors.append("Poor baseline function")
        elif baseline_function < 60:
            risk_score += 8
            risk_factors.append("Reduced baseline function")
        
        # Risk categorization
        if risk_score <= 20:
            risk_category = "low"
        elif risk_score <= 40:
            risk_category = "moderate"
        elif risk_score <= 60:
            risk_category = "high"
        else:
            risk_category = "very_high"
        
        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "risk_factors": risk_factors,
            "protective_factors": RecoveryMetricsCalculator._identify_protective_factors(age, bmi, baseline_function),
            "recommendations": RecoveryMetricsCalculator._generate_risk_recommendations(risk_category, risk_factors)
        }
    
    @staticmethod
    def _identify_protective_factors(age: int, bmi: float, baseline_function: float) -> List[str]:
        """Identify protective factors for recovery"""
        protective_factors = []
        
        if age < 45:
            protective_factors.append("Young age (<45)")
        
        if 18.5 <= bmi <= 24.9:
            protective_factors.append("Normal BMI")
        
        if baseline_function >= 80:
            protective_factors.append("Excellent baseline function")
        elif baseline_function >= 70:
            protective_factors.append("Good baseline function")
        
        return protective_factors
    
    @staticmethod
    def _generate_risk_recommendations(risk_category: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        if risk_category in ["high", "very_high"]:
            recommendations.extend([
                "Consider enhanced monitoring protocol",
                "Early physical therapy consultation",
                "Nutritional assessment if BMI elevated",
                "Pain management optimization"
            ])
        
        if any("diabetes" in factor.lower() for factor in risk_factors):
            recommendations.append("Glucose management optimization")
        
        if any("obesity" in factor.lower() for factor in risk_factors):
            recommendations.append("Weight management program referral")
        
        if any("depression" in factor.lower() for factor in risk_factors):
            recommendations.append("Mental health support consideration")
        
        if not recommendations:
            recommendations.append("Standard care protocol appropriate")
        
        return recommendations


class TrendAnalyzer:
    """Analyzer for metric trends and patterns"""
    
    @staticmethod
    def analyze_metric_trend(
        values: List[float],
        dates: List[date],
        metric_name: str,
        smoothing_window: int = 7
    ) -> MetricTrend:
        """
        Analyze trend in a metric over time
        """
        if len(values) < 3 or len(values) != len(dates):
            return MetricTrend(
                metric_name=metric_name,
                trend_direction="insufficient_data",
                trend_magnitude=0.0,
                trend_significance=0.0,
                trend_velocity=0.0,
                confidence_level="low"
            )
        
        # Apply smoothing if enough data points
        if len(values) >= smoothing_window:
            smoothed_values = TrendAnalyzer._apply_moving_average(values, smoothing_window)
        else:
            smoothed_values = values
        
        # Calculate linear trend
        x = np.array(range(len(smoothed_values)))
        y = np.array(smoothed_values)
        
        try:
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if slope > 0.1 and r_squared > 0.3:
                direction = "improving"
            elif slope < -0.1 and r_squared > 0.3:
                direction = "declining"
            else:
                direction = "stable"
            
            # Calculate magnitude (standardized slope)
            magnitude = abs(slope) / (np.std(y) + 1e-6)
            
            # Calculate velocity (change per day)
            days_span = (dates[-1] - dates[0]).days
            velocity = slope / max(days_span, 1) if days_span > 0 else 0
            
            # Determine confidence
            if r_squared > 0.7:
                confidence = "high"
            elif r_squared > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return MetricTrend(
                metric_name=metric_name,
                trend_direction=direction,
                trend_magnitude=magnitude,
                trend_significance=r_squared,
                trend_velocity=velocity,
                confidence_level=confidence
            )
            
        except Exception:
            return MetricTrend(
                metric_name=metric_name,
                trend_direction="error",
                trend_magnitude=0.0,
                trend_significance=0.0,
                trend_velocity=0.0,
                confidence_level="low"
            )
    
    @staticmethod
    def _apply_moving_average(values: List[float], window: int) -> List[float]:
        """Apply moving average smoothing"""
        if window >= len(values):
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(values), i + window // 2 + 1)
            window_values = values[start_idx:end_idx]
            smoothed.append(np.mean(window_values))
        
        return smoothed
    
    @staticmethod
    def detect_changepoints(values: List[float], min_segment_length: int = 5) -> List[int]:
        """
        Detect significant change points in time series data
        Simple implementation using variance change detection
        """
        if len(values) < min_segment_length * 2:
            return []
        
        changepoints = []
        
        for i in range(min_segment_length, len(values) - min_segment_length):
            # Calculate variance before and after potential changepoint
            before_segment = values[max(0, i - min_segment_length):i]
            after_segment = values[i:min(len(values), i + min_segment_length)]
            
            if len(before_segment) >= 3 and len(after_segment) >= 3:
                var_before = np.var(before_segment)
                var_after = np.var(after_segment)
                mean_before = np.mean(before_segment)
                mean_after = np.mean(after_segment)
                
                # Check for significant mean change
                mean_change = abs(mean_after - mean_before)
                pooled_std = np.sqrt((var_before + var_after) / 2)
                
                # If mean change is > 2 standard deviations, consider it a changepoint
                if mean_change > 2 * pooled_std and pooled_std > 0:
                    changepoints.append(i)
        
        return changepoints
    
    @staticmethod
    def analyze_cyclical_patterns(
        values: List[float], 
        dates: List[date],
        expected_cycle_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze cyclical patterns (e.g., weekly patterns in activity)
        """
        if len(values) < expected_cycle_days * 2:
            return {"has_cycle": False, "reason": "insufficient_data"}
        
        # Group by day of cycle
        cycle_groups = [[] for _ in range(expected_cycle_days)]
        
        for i, (value, date_val) in enumerate(zip(values, dates)):
            day_of_cycle = i % expected_cycle_days
            cycle_groups[day_of_cycle].append(value)
        
        # Calculate statistics for each day of cycle
        cycle_stats = []
        for day_group in cycle_groups:
            if day_group:
                cycle_stats.append({
                    "mean": np.mean(day_group),
                    "std": np.std(day_group),
                    "count": len(day_group)
                })
            else:
                cycle_stats.append({"mean": 0, "std": 0, "count": 0})
        
        # Calculate cycle strength (variance between days / variance within days)
        means = [stat["mean"] for stat in cycle_stats if stat["count"] > 0]
        
        if len(means) < expected_cycle_days // 2:
            return {"has_cycle": False, "reason": "sparse_data"}
        
        between_variance = np.var(means) if len(means) > 1 else 0
        
        # Average within-day variance
        within_variances = [stat["std"]**2 for stat in cycle_stats if stat["count"] > 1]
        avg_within_variance = np.mean(within_variances) if within_variances else 1
        
        # Cycle strength ratio
        cycle_strength = between_variance / (avg_within_variance + 1e-6)
        
        return {
            "has_cycle": cycle_strength > 1.5,  # Threshold for significant cycle
            "cycle_strength": cycle_strength,
            "cycle_period_days": expected_cycle_days,
            "day_patterns": cycle_stats,
            "peak_days": [i for i, stat in enumerate(cycle_stats) if stat["mean"] == max(s["mean"] for s in cycle_stats)],
            "low_days": [i for i, stat in enumerate(cycle_stats) if stat["mean"] == min(s["mean"] for s in cycle_stats)]
        }


class MilestoneTracker:
    """Tracker for recovery milestones"""
    
    @staticmethod
    def create_diagnosis_milestones(diagnosis_type: str, surgery_date: date) -> List[RecoveryMilestone]:
        """Create diagnosis-specific recovery milestones"""
        milestones = []
        
        if "ACL" in diagnosis_type:
            milestones.extend([
                RecoveryMilestone(
                    milestone_id="acl_week2_swelling",
                    name="Swelling Control",
                    description="Minimal knee swelling",
                    target_week=2,
                    metric_type=MetricCategory.FUNCTION,
                    target_value=60,
                    tolerance_range=(50, 70),
                    critical=True,
                    achieved=False,
                    achievement_date=None
                ),
                RecoveryMilestone(
                    milestone_id="acl_week6_mobility",
                    name="Range of Motion",
                    description="Near full knee flexion (120°)",
                    target_week=6,
                    metric_type=MetricCategory.MOBILITY,
                    target_value=120,
                    tolerance_range=(110, 130),
                    critical=True,
                    achieved=False,
                    achievement_date=None
                ),
                RecoveryMilestone(
                    milestone_id="acl_week12_strength",
                    name="Strength Recovery",
                    description="70% strength compared to uninjured leg",
                    target_week=12,
                    metric_type=MetricCategory.FUNCTION,
                    target_value=70,
                    tolerance_range=(65, 80),
                    critical=False,
                    achieved=False,
                    achievement_date=None
                )
            ])
        
        elif "Rotator Cuff" in diagnosis_type:
            milestones.extend([
                RecoveryMilestone(
                    milestone_id="rc_week6_passive_rom",
                    name="Passive Range of Motion",
                    description="Passive forward flexion to 140°",
                    target_week=6,
                    metric_type=MetricCategory.MOBILITY,
                    target_value=140,
                    tolerance_range=(130, 150),
                    critical=True,
                    achieved=False,
                    achievement_date=None
                ),
                RecoveryMilestone(
                    milestone_id="rc_week12_active_rom",
                    name="Active Range of Motion",
                    description="Active forward flexion to 160°",
                    target_week=12,
                    metric_type=MetricCategory.MOBILITY,
                    target_value=160,
                    tolerance_range=(150, 170),
                    critical=True,
                    achieved=False,
                    achievement_date=None
                )
            ])
        
        # Add common milestones for all diagnoses
        milestones.extend([
            RecoveryMilestone(
                milestone_id="common_week1_pain",
                name="Pain Management",
                description="Pain controlled (≤4/10)",
                target_week=1,
                metric_type=MetricCategory.PAIN,
                target_value=6,  # KOOS/ASES style (higher = less pain)
                tolerance_range=(4, 8),
                critical=True,
                achieved=False,
                achievement_date=None
            ),
            RecoveryMilestone(
                milestone_id="common_week8_activity",
                name="Activity Level",
                description="5000 daily steps consistently",
                target_week=8,
                metric_type=MetricCategory.ACTIVITY,
                target_value=5000,
                tolerance_range=(4000, 6000),
                critical=False,
                achieved=False,
                achievement_date=None
            )
        ])
        
        return milestones
    
    @staticmethod
    def check_milestone_achievement(
        milestone: RecoveryMilestone,
        current_value: float,
        current_date: date
    ) -> bool:
        """Check if a milestone has been achieved"""
        if milestone.achieved:
            return True
        
        # Check if value is within tolerance range
        min_val, max_val = milestone.tolerance_range
        
        if min_val <= current_value <= max_val:
            milestone.achieved = True
            milestone.achievement_date = current_date
            return True
        
        return False
    
    @staticmethod
    def calculate_milestone_progress(
        milestones: List[RecoveryMilestone],
        weeks_post_surgery: int
    ) -> Dict[str, Any]:
        """Calculate overall milestone progress"""
        total_milestones = len(milestones)
        achieved_milestones = sum(1 for m in milestones if m.achieved)
        
        # Milestones that should have been achieved by now
        due_milestones = [m for m in milestones if m.target_week <= weeks_post_surgery]
        achieved_due = sum(1 for m in due_milestones if m.achieved)
        
        # Critical milestones
        critical_milestones = [m for m in milestones if m.critical]
        critical_achieved = sum(1 for m in critical_milestones if m.achieved)
        critical_due = [m for m in critical_milestones if m.target_week <= weeks_post_surgery]
        critical_due_achieved = sum(1 for m in critical_due if m.achieved)
        
        # Progress scores
        overall_progress = (achieved_milestones / total_milestones * 100) if total_milestones > 0 else 0
        timely_progress = (achieved_due / len(due_milestones) * 100) if due_milestones else 0
        critical_progress = (critical_achieved / len(critical_milestones) * 100) if critical_milestones else 0
        
        return {
            "overall_progress": overall_progress,
            "timely_progress": timely_progress,
            "critical_progress": critical_progress,
            "total_milestones": total_milestones,
            "achieved_milestones": achieved_milestones,
            "due_milestones": len(due_milestones),
            "achieved_due": achieved_due,
            "critical_milestones": len(critical_milestones),
            "critical_achieved": critical_achieved,
            "missed_critical": [m for m in critical_due if not m.achieved],
            "upcoming_milestones": [m for m in milestones if not m.achieved and m.target_week > weeks_post_surgery]
        }