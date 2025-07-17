"""
Predictive analytics for recovery outcomes and risk assessment.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from recovery_trajectories import (
    DiagnosisType, 
    get_expected_score_at_week,
    calculate_weeks_post_surgery,
    is_score_within_corridor
)

class RecoveryPredictor:
    """Predictive analytics for recovery outcomes"""
    
    @staticmethod
    def calculate_risk_score(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        previous_scores: Optional[Dict[str, float]] = None,
        wearable_data: Optional[Dict] = None,
        missed_assessments: int = 0
    ) -> float:
        """
        Calculate comprehensive risk score (0-100)
        Higher scores indicate higher risk of poor outcomes
        """
        risk_factors = []
        
        # Base risk by diagnosis type
        base_risk = {
            DiagnosisType.ACL_TEAR: 25,
            DiagnosisType.MENISCUS_TEAR: 15,
            DiagnosisType.CARTILAGE_DEFECT: 30,
            DiagnosisType.KNEE_OSTEOARTHRITIS: 20,
            DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT: 20,
            DiagnosisType.ROTATOR_CUFF_TEAR: 35,
            DiagnosisType.LABRAL_TEAR: 20,
            DiagnosisType.SHOULDER_INSTABILITY: 25,
            DiagnosisType.SHOULDER_OSTEOARTHRITIS: 25,
            DiagnosisType.POST_TOTAL_SHOULDER_REPLACEMENT: 25,
        }
        
        risk_score = base_risk.get(diagnosis, 25)
        
        # Trajectory deviation analysis
        trajectory_risk = RecoveryPredictor._calculate_trajectory_risk(
            diagnosis, weeks_post_surgery, current_scores
        )
        risk_score += trajectory_risk
        
        # Recovery velocity analysis
        if previous_scores:
            velocity_risk = RecoveryPredictor._calculate_velocity_risk(
                current_scores, previous_scores
            )
            risk_score += velocity_risk
        
        # Wearable data analysis
        if wearable_data:
            wearable_risk = RecoveryPredictor._calculate_wearable_risk(wearable_data)
            risk_score += wearable_risk
        
        # Missed assessment penalty
        assessment_risk = min(missed_assessments * 5, 20)
        risk_score += assessment_risk
        
        return min(max(risk_score, 0), 100)
    
    @staticmethod
    def _calculate_trajectory_risk(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float]
    ) -> float:
        """Calculate risk based on trajectory deviation"""
        risk_points = 0
        
        subscales = list(current_scores.keys())
        
        for subscale in subscales:
            expected_point = get_expected_score_at_week(diagnosis, weeks_post_surgery, subscale)
            if expected_point:
                actual_score = current_scores[subscale]
                
                # Calculate deviation from expected
                deviation = expected_point.expected_score - actual_score
                
                # Severe deviation (>20 points below expected)
                if deviation > 20:
                    risk_points += 15
                # Moderate deviation (10-20 points below expected)
                elif deviation > 10:
                    risk_points += 8
                # Minor deviation (5-10 points below expected)
                elif deviation > 5:
                    risk_points += 3
        
        return min(risk_points, 30)
    
    @staticmethod
    def _calculate_velocity_risk(
        current_scores: Dict[str, float],
        previous_scores: Dict[str, float]
    ) -> float:
        """Calculate risk based on recovery velocity"""
        risk_points = 0
        declining_subscales = 0
        stagnant_subscales = 0
        
        for subscale in current_scores.keys():
            if subscale in previous_scores:
                change = current_scores[subscale] - previous_scores[subscale]
                
                # Declining scores
                if change < -5:
                    declining_subscales += 1
                    risk_points += 8
                # Stagnant scores
                elif abs(change) < 2:
                    stagnant_subscales += 1
                    risk_points += 3
        
        # Additional risk for multiple declining subscales
        if declining_subscales >= 2:
            risk_points += 10
        
        return min(risk_points, 25)
    
    @staticmethod
    def _calculate_wearable_risk(wearable_data: Dict) -> float:
        """Calculate risk based on wearable data trends"""
        risk_points = 0
        
        steps = wearable_data.get('steps', 0)
        sleep_hours = wearable_data.get('sleep_hours', 0)
        heart_rate = wearable_data.get('heart_rate', 0)
        
        # Very low activity
        if steps < 1000:
            risk_points += 10
        elif steps < 2000:
            risk_points += 5
        
        # Poor sleep
        if sleep_hours < 5:
            risk_points += 8
        elif sleep_hours < 6:
            risk_points += 4
        
        # Elevated resting heart rate (potential stress/inflammation)
        if heart_rate > 90:
            risk_points += 5
        
        return min(risk_points, 15)
    
    @staticmethod
    def get_risk_category(risk_score: float) -> str:
        """Categorize risk score"""
        if risk_score < 25:
            return "Low"
        elif risk_score < 50:
            return "Moderate"
        elif risk_score < 75:
            return "High"
        else:
            return "Very High"
    
    @staticmethod
    def calculate_recovery_velocity(
        current_scores: Dict[str, float],
        previous_scores: Optional[Dict[str, float]] = None,
        weeks_between: int = 2
    ) -> List[Dict]:
        """Calculate recovery velocity for each subscale"""
        velocity_data = []
        
        for subscale, current_score in current_scores.items():
            velocity_info = {
                "subscale": subscale,
                "current_score": current_score,
                "previous_score": None,
                "velocity": None,
                "trend": "Unknown"
            }
            
            if previous_scores and subscale in previous_scores:
                previous_score = previous_scores[subscale]
                velocity_info["previous_score"] = previous_score
                
                # Calculate velocity (points per week)
                score_change = current_score - previous_score
                velocity = score_change / weeks_between if weeks_between > 0 else 0
                velocity_info["velocity"] = velocity
                
                # Determine trend
                if velocity > 2:
                    velocity_info["trend"] = "Improving"
                elif velocity < -2:
                    velocity_info["trend"] = "Declining"
                else:
                    velocity_info["trend"] = "Stable"
            
            velocity_data.append(velocity_info)
        
        return velocity_data
    
    @staticmethod
    def predict_recovery_timeline(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        recovery_velocity: List[Dict],
        target_score: float = 85
    ) -> Tuple[Optional[datetime], Optional[str]]:
        """
        Predict when patient will reach target recovery score
        Returns (projected_date, confidence_interval)
        """
        # Calculate average velocity for improving subscales
        improving_velocities = [
            v["velocity"] for v in recovery_velocity 
            if v["velocity"] and v["velocity"] > 0
        ]
        
        if not improving_velocities:
            return None, None
        
        avg_velocity = sum(improving_velocities) / len(improving_velocities)
        
        # Calculate current average score
        current_avg = sum(current_scores.values()) / len(current_scores)
        
        # Calculate weeks needed to reach target
        score_gap = target_score - current_avg
        
        if score_gap <= 0:
            # Already at target
            return datetime.now(), "±0 weeks"
        
        if avg_velocity <= 0:
            # No improvement trend
            return None, None
        
        weeks_needed = score_gap / avg_velocity
        
        # Add current weeks post-surgery
        total_weeks = weeks_post_surgery + weeks_needed
        
        # Calculate projected date
        projected_date = datetime.now() + timedelta(weeks=weeks_needed)
        
        # Calculate confidence interval based on velocity consistency
        velocity_std = RecoveryPredictor._calculate_velocity_std(improving_velocities)
        confidence_weeks = max(1, velocity_std * 2)  # 2 standard deviations
        
        confidence_interval = f"±{confidence_weeks:.0f} weeks"
        
        return projected_date, confidence_interval
    
    @staticmethod
    def _calculate_velocity_std(velocities: List[float]) -> float:
        """Calculate standard deviation of velocities"""
        if len(velocities) < 2:
            return 1.0
        
        mean_velocity = sum(velocities) / len(velocities)
        variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)
        return math.sqrt(variance)
    
    @staticmethod
    def detect_concerning_patterns(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        recovery_velocity: List[Dict],
        milestone_status: List[Dict]
    ) -> List[str]:
        """Detect concerning recovery patterns"""
        patterns = []
        
        # Plateau detection (no improvement in multiple subscales)
        stagnant_count = sum(1 for v in recovery_velocity if v["trend"] == "Stable")
        if stagnant_count >= len(recovery_velocity) / 2:
            patterns.append("Multiple subscales showing plateau pattern")
        
        # Regression detection
        declining_count = sum(1 for v in recovery_velocity if v["trend"] == "Declining")
        if declining_count >= 2:
            patterns.append("Declining scores in multiple areas")
        
        # Subscale imbalance detection
        scores = list(current_scores.values())
        if len(scores) >= 3:
            score_range = max(scores) - min(scores)
            if score_range > 30:
                patterns.append("Significant imbalance between recovery domains")
        
        # Missed critical milestones
        missed_critical = [
            m for m in milestone_status 
            if m["critical"] and not m["achieved"] and m["week"] <= weeks_post_surgery
        ]
        if missed_critical:
            patterns.append(f"Critical milestone missed: {missed_critical[0]['description']}")
        
        # Chronic pain pattern (consistently low pain scores)
        if "pain_score" in current_scores and current_scores["pain_score"] < 50:
            patterns.append("Persistent pain affecting recovery")
        elif "pain_component" in current_scores and current_scores["pain_component"] < 25:
            patterns.append("Persistent pain affecting recovery")
        
        return patterns
    
    @staticmethod
    def identify_positive_trends(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        recovery_velocity: List[Dict],
        milestone_status: List[Dict]
    ) -> List[str]:
        """Identify positive recovery trends"""
        trends = []
        
        # Consistent improvement across subscales
        improving_count = sum(1 for v in recovery_velocity if v["trend"] == "Improving")
        if improving_count >= len(recovery_velocity) / 2:
            trends.append("Consistent improvement across multiple domains")
        
        # Ahead of schedule in key areas
        ahead_schedule = [
            subscale for subscale, score in current_scores.items()
            if RecoveryPredictor._is_ahead_of_schedule(diagnosis, weeks_post_surgery, subscale, score)
        ]
        if ahead_schedule:
            trends.append(f"Ahead of schedule in {', '.join(ahead_schedule)}")
        
        # Milestone achievements
        recent_achievements = [
            m for m in milestone_status 
            if m["achieved"] and m["week"] >= weeks_post_surgery - 2
        ]
        if recent_achievements:
            trends.append(f"Recent milestone achieved: {recent_achievements[0]['description']}")
        
        # High velocity improvement
        high_velocity = [
            v for v in recovery_velocity 
            if v["velocity"] and v["velocity"] > 3
        ]
        if high_velocity:
            trends.append("Rapid improvement in key recovery areas")
        
        return trends
    
    @staticmethod
    def _is_ahead_of_schedule(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        subscale: str,
        actual_score: float
    ) -> bool:
        """Check if score is ahead of expected trajectory"""
        expected_point = get_expected_score_at_week(diagnosis, weeks_post_surgery, subscale)
        return expected_point and actual_score > expected_point.upper_bound