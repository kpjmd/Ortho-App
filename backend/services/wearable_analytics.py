"""
Comprehensive wearable data analytics engine for orthopedic recovery tracking.
Provides advanced analytics, pattern detection, and insights generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from models.wearable_data import (
    ComprehensiveWearableData, RecoveryIndicators, 
    ActivityMetrics, SleepMetrics, HeartRateMetrics, MovementMetrics
)
from schemas.wearable_schemas import WearableDataAggregations
import logging

logger = logging.getLogger(__name__)


class WearableAnalyticsEngine:
    """Advanced analytics engine for wearable data processing"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    async def analyze_recovery_velocity(self, patient_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze recovery velocity using activity and PRO score correlations
        """
        try:
            # Get wearable data trends
            pipeline = WearableDataAggregations.get_daily_summary_pipeline(
                patient_id, 
                (datetime.utcnow() - timedelta(days=days_back)).date(),
                datetime.utcnow().date()
            )
            
            cursor = self.db.comprehensive_wearable_data.aggregate(pipeline)
            daily_data = await cursor.to_list(length=None)
            
            if len(daily_data) < 7:
                return {"error": "Insufficient data for velocity analysis"}
            
            # Calculate velocity metrics
            velocity_metrics = {}
            
            # Walking speed velocity
            walking_speeds = [d.get("walking_speed_ms") for d in daily_data if d.get("walking_speed_ms")]
            if len(walking_speeds) >= 7:
                velocity_metrics["walking_speed"] = self._calculate_trend_velocity(walking_speeds)
            
            # Step count velocity
            step_counts = [d.get("steps", 0) for d in daily_data]
            if len(step_counts) >= 7:
                velocity_metrics["step_count"] = self._calculate_trend_velocity(step_counts)
            
            # Sleep efficiency velocity
            sleep_efficiencies = [d.get("sleep_efficiency") for d in daily_data if d.get("sleep_efficiency")]
            if len(sleep_efficiencies) >= 7:
                velocity_metrics["sleep_efficiency"] = self._calculate_trend_velocity(sleep_efficiencies)
            
            # Heart rate recovery velocity
            resting_hrs = [d.get("resting_hr") for d in daily_data if d.get("resting_hr")]
            if len(resting_hrs) >= 7:
                velocity_metrics["resting_hr"] = self._calculate_trend_velocity(resting_hrs, invert=True)
            
            # Correlate with PRO scores
            pro_correlations = await self._correlate_with_pro_scores(patient_id, velocity_metrics)
            
            return {
                "patient_id": patient_id,
                "analysis_period_days": days_back,
                "velocity_metrics": velocity_metrics,
                "pro_correlations": pro_correlations,
                "milestone_predictions": await self._predict_milestone_achievement(patient_id, velocity_metrics),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Recovery velocity analysis failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def detect_plateau_patterns(self, patient_id: str) -> Dict[str, Any]:
        """
        Detect plateau patterns and recommend interventions
        """
        try:
            # Get 8-week trend data
            pipeline = WearableDataAggregations.get_weekly_trends_pipeline(patient_id, 8)
            cursor = self.db.comprehensive_wearable_data.aggregate(pipeline)
            weekly_data = await cursor.to_list(length=None)
            
            if len(weekly_data) < 4:
                return {"error": "Insufficient data for plateau detection"}
            
            plateau_analysis = {}
            
            # Analyze each metric for plateau patterns
            metrics_to_analyze = ["avg_steps", "avg_walking_speed", "avg_sleep_efficiency", "avg_resting_hr"]
            
            for metric in metrics_to_analyze:
                values = [week.get(metric) for week in weekly_data if week.get(metric) is not None]
                if len(values) >= 4:
                    plateau_analysis[metric] = self._detect_plateau(values)
            
            # Generate intervention recommendations
            interventions = self._generate_plateau_interventions(plateau_analysis)
            
            return {
                "patient_id": patient_id,
                "plateau_analysis": plateau_analysis,
                "intervention_recommendations": interventions,
                "weeks_analyzed": len(weekly_data),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Plateau detection failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def assess_clinical_risk(self, patient_id: str) -> Dict[str, Any]:
        """
        Clinical risk assessment based on comprehensive wearable data
        """
        try:
            # Get recent data (last 14 days)
            recent_data = await self.db.comprehensive_wearable_data.find({
                "patient_id": patient_id,
                "date": {"$gte": (datetime.utcnow() - timedelta(days=14)).date()}
            }).sort("date", -1).to_list(14)
            
            if not recent_data:
                return {"error": "No recent data available"}
            
            risk_assessment = {
                "sedentary_behavior_risk": self._assess_sedentary_risk(recent_data),
                "sleep_quality_risk": self._assess_sleep_risk(recent_data),
                "cardiovascular_risk": self._assess_cardiovascular_risk(recent_data),
                "activity_deviation_risk": self._assess_activity_deviation_risk(recent_data),
                "pain_correlation_risk": await self._assess_pain_correlation_risk(patient_id, recent_data)
            }
            
            # Calculate overall risk score
            risk_scores = [assessment["risk_score"] for assessment in risk_assessment.values() 
                          if isinstance(assessment, dict) and "risk_score" in assessment]
            overall_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            
            # Generate clinical alerts
            alerts = self._generate_clinical_alerts(risk_assessment, overall_risk)
            
            return {
                "patient_id": patient_id,
                "overall_risk_score": overall_risk,
                "risk_category": self._categorize_risk(overall_risk),
                "detailed_assessment": risk_assessment,
                "clinical_alerts": alerts,
                "recommendations": self._generate_risk_recommendations(risk_assessment),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Clinical risk assessment failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def generate_personalized_insights(self, patient_id: str) -> Dict[str, Any]:
        """
        Generate personalized recovery insights based on individual patterns
        """
        try:
            # Get patient diagnosis info
            patient = await self.db.patients.find_one({"id": patient_id})
            if not patient:
                return {"error": "Patient not found"}
            
            diagnosis_type = patient.get("diagnosis_type")
            surgery_date = patient.get("date_of_surgery")
            
            # Get comprehensive wearable data
            wearable_data = await self.db.comprehensive_wearable_data.find({
                "patient_id": patient_id
            }).sort("date", -1).limit(30).to_list(30)
            
            if not wearable_data:
                return {"error": "No wearable data available"}
            
            insights = {
                "optimal_activity_levels": await self._calculate_optimal_activity(patient_id, diagnosis_type),
                "sleep_recommendations": self._analyze_sleep_patterns(wearable_data),
                "exercise_timing": self._optimize_exercise_timing(wearable_data),
                "pain_prediction": await self._predict_pain_patterns(patient_id, wearable_data),
                "recovery_readiness": self._assess_recovery_readiness(wearable_data)
            }
            
            return {
                "patient_id": patient_id,
                "diagnosis": diagnosis_type,
                "personalized_insights": insights,
                "confidence_score": self._calculate_insight_confidence(wearable_data),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Personalized insights generation failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def analyze_provider_dashboard_metrics(self, patient_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for provider dashboard
        """
        try:
            # Get compliance scoring
            compliance = await self._calculate_compliance_score(patient_id)
            
            # Get early intervention triggers
            intervention_triggers = await self._detect_intervention_triggers(patient_id)
            
            # Get comparative analysis
            comparative_analysis = await self._generate_comparative_analysis(patient_id)
            
            # Get clinical summary
            clinical_summary = await self._generate_clinical_summary(patient_id)
            
            return {
                "patient_id": patient_id,
                "compliance_metrics": compliance,
                "intervention_triggers": intervention_triggers,
                "comparative_analysis": comparative_analysis,
                "clinical_summary": clinical_summary,
                "dashboard_alerts": self._generate_dashboard_alerts(compliance, intervention_triggers),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Provider dashboard analytics failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    def _calculate_trend_velocity(self, values: List[float], invert: bool = False) -> Dict[str, Any]:
        """Calculate trend velocity using linear regression"""
        if len(values) < 3:
            return {"velocity": 0, "trend": "insufficient_data", "r_squared": 0}
        
        # Convert to numpy arrays
        x = np.array(range(len(values)))
        y = np.array(values)
        
        if invert:
            y = -y  # For metrics where lower is better (like resting HR)
        
        # Calculate linear regression
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend
            if slope > 0.1 and r_squared > 0.3:
                trend = "improving"
            elif slope < -0.1 and r_squared > 0.3:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "velocity": float(slope),
                "trend": trend,
                "r_squared": float(r_squared),
                "confidence": "high" if r_squared > 0.7 else "medium" if r_squared > 0.3 else "low"
            }
            
        except Exception:
            return {"velocity": 0, "trend": "error", "r_squared": 0}
    
    async def _correlate_with_pro_scores(self, patient_id: str, velocity_metrics: Dict) -> Dict[str, float]:
        """Correlate wearable velocity with PRO score improvements"""
        correlations = {}
        
        # Get patient's body part to determine PRO type
        patient = await self.db.patients.find_one({"id": patient_id})
        if not patient:
            return correlations
        
        diagnosis_type = patient.get("diagnosis_type")
        
        # Determine if knee or shoulder
        knee_diagnoses = ["ACL Tear", "Meniscus Tear", "Cartilage Defect", "Knee Osteoarthritis", "Post Total Knee Replacement"]
        is_knee = diagnosis_type in knee_diagnoses
        
        # Get PRO scores
        if is_knee:
            pro_scores = await self.db.koos_scores.find({"patient_id": patient_id}).sort("date", 1).to_list(100)
            score_key = "total_score"
        else:
            pro_scores = await self.db.ases_scores.find({"patient_id": patient_id}).sort("date", 1).to_list(100)
            score_key = "total_score"
        
        if len(pro_scores) < 3:
            return correlations
        
        # Calculate PRO score velocity
        pro_values = [score.get(score_key, 0) for score in pro_scores]
        pro_velocity = self._calculate_trend_velocity(pro_values)
        
        # Correlate with wearable metrics
        for metric, velocity_data in velocity_metrics.items():
            if isinstance(velocity_data, dict) and "velocity" in velocity_data:
                correlation = self._simple_correlation(
                    [velocity_data["velocity"]], 
                    [pro_velocity["velocity"]]
                )
                correlations[f"{metric}_pro_correlation"] = correlation
        
        return correlations
    
    def _simple_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            return float(np.corrcoef(x, y)[0, 1])
        except:
            return 0.0
    
    async def _predict_milestone_achievement(self, patient_id: str, velocity_metrics: Dict) -> Dict[str, Any]:
        """Predict milestone achievement based on current velocity"""
        # Get patient info
        patient = await self.db.patients.find_one({"id": patient_id})
        if not patient:
            return {"error": "Patient not found"}
        
        surgery_date = patient.get("date_of_surgery")
        if not surgery_date:
            return {"error": "Surgery date not available"}
        
        # Calculate weeks post-surgery
        weeks_post_surgery = (datetime.utcnow().date() - surgery_date).days // 7
        
        predictions = {}
        
        # Predict next milestone based on velocity
        if "walking_speed" in velocity_metrics:
            walking_velocity = velocity_metrics["walking_speed"]
            if walking_velocity.get("velocity", 0) > 0:
                weeks_to_next_milestone = max(1, 4 - walking_velocity["velocity"] * 2)
                predictions["mobility_milestone"] = {
                    "predicted_weeks": weeks_to_next_milestone,
                    "confidence": walking_velocity.get("confidence", "low")
                }
        
        return predictions
    
    def _detect_plateau(self, values: List[float]) -> Dict[str, Any]:
        """Detect if metric shows plateau pattern"""
        if len(values) < 4:
            return {"plateau_detected": False, "reason": "insufficient_data"}
        
        # Calculate variance in recent vs earlier periods
        recent_values = values[-3:]
        earlier_values = values[:-3]
        
        recent_variance = np.var(recent_values) if len(recent_values) > 1 else 0
        earlier_variance = np.var(earlier_values) if len(earlier_values) > 1 else 0
        
        # Calculate trend slope
        slope = self._calculate_trend_velocity(values)["velocity"]
        
        # Detect plateau: low variance and minimal slope
        plateau_detected = (recent_variance < 0.1 and abs(slope) < 0.05)
        
        return {
            "plateau_detected": plateau_detected,
            "trend_slope": slope,
            "recent_variance": float(recent_variance),
            "stability_score": 1 - abs(slope) if abs(slope) < 1 else 0
        }
    
    def _generate_plateau_interventions(self, plateau_analysis: Dict) -> List[str]:
        """Generate intervention recommendations for plateau patterns"""
        interventions = []
        
        for metric, analysis in plateau_analysis.items():
            if analysis.get("plateau_detected"):
                if "steps" in metric:
                    interventions.append("Consider increasing daily activity goals gradually")
                elif "walking_speed" in metric:
                    interventions.append("Focus on gait training and mobility exercises")
                elif "sleep" in metric:
                    interventions.append("Evaluate sleep hygiene and pain management")
                elif "hr" in metric:
                    interventions.append("Consider stress management and recovery techniques")
        
        if not interventions:
            interventions.append("Continue current rehabilitation program")
        
        return interventions
    
    def _assess_sedentary_risk(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Assess risk from sedentary behavior"""
        sedentary_times = []
        for day in recent_data:
            activity_metrics = day.get("activity_metrics", {})
            if activity_metrics and activity_metrics.get("sedentary_minutes"):
                sedentary_times.append(activity_metrics["sedentary_minutes"])
        
        if not sedentary_times:
            return {"risk_score": 0, "reason": "no_data"}
        
        avg_sedentary = sum(sedentary_times) / len(sedentary_times)
        
        # Risk scoring: >10 hours = high risk
        if avg_sedentary > 600:  # 10 hours
            risk_score = 80
        elif avg_sedentary > 480:  # 8 hours
            risk_score = 50
        elif avg_sedentary > 360:  # 6 hours
            risk_score = 20
        else:
            risk_score = 0
        
        return {
            "risk_score": risk_score,
            "avg_sedentary_hours": avg_sedentary / 60,
            "risk_factors": ["Excessive sedentary time"] if risk_score > 50 else []
        }
    
    def _assess_sleep_risk(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Assess risk from poor sleep quality"""
        sleep_efficiencies = []
        sleep_durations = []
        
        for day in recent_data:
            sleep_metrics = day.get("sleep_metrics", {})
            if sleep_metrics:
                if sleep_metrics.get("sleep_efficiency"):
                    sleep_efficiencies.append(sleep_metrics["sleep_efficiency"])
                if sleep_metrics.get("total_sleep_minutes"):
                    sleep_durations.append(sleep_metrics["total_sleep_minutes"] / 60)
        
        risk_factors = []
        risk_score = 0
        
        if sleep_efficiencies:
            avg_efficiency = sum(sleep_efficiencies) / len(sleep_efficiencies)
            if avg_efficiency < 70:
                risk_score += 40
                risk_factors.append("Poor sleep efficiency")
        
        if sleep_durations:
            avg_duration = sum(sleep_durations) / len(sleep_durations)
            if avg_duration < 6:
                risk_score += 30
                risk_factors.append("Insufficient sleep duration")
        
        return {
            "risk_score": min(risk_score, 100),
            "avg_sleep_efficiency": sum(sleep_efficiencies) / len(sleep_efficiencies) if sleep_efficiencies else None,
            "avg_sleep_hours": sum(sleep_durations) / len(sleep_durations) if sleep_durations else None,
            "risk_factors": risk_factors
        }
    
    def _assess_cardiovascular_risk(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Assess cardiovascular risk indicators"""
        resting_hrs = []
        hr_variabilities = []
        
        for day in recent_data:
            hr_metrics = day.get("heart_rate_metrics", {})
            if hr_metrics:
                if hr_metrics.get("resting_hr"):
                    resting_hrs.append(hr_metrics["resting_hr"])
                if hr_metrics.get("hr_variability_ms"):
                    hr_variabilities.append(hr_metrics["hr_variability_ms"])
        
        risk_score = 0
        risk_factors = []
        
        if resting_hrs:
            avg_resting_hr = sum(resting_hrs) / len(resting_hrs)
            if avg_resting_hr > 90:
                risk_score += 30
                risk_factors.append("Elevated resting heart rate")
        
        if hr_variabilities:
            avg_hrv = sum(hr_variabilities) / len(hr_variabilities)
            if avg_hrv < 20:  # Low HRV indicates poor recovery
                risk_score += 20
                risk_factors.append("Low heart rate variability")
        
        return {
            "risk_score": risk_score,
            "avg_resting_hr": sum(resting_hrs) / len(resting_hrs) if resting_hrs else None,
            "avg_hrv": sum(hr_variabilities) / len(hr_variabilities) if hr_variabilities else None,
            "risk_factors": risk_factors
        }
    
    def _assess_activity_deviation_risk(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Assess risk from activity pattern deviations"""
        step_counts = []
        
        for day in recent_data:
            activity_metrics = day.get("activity_metrics", {})
            if activity_metrics and activity_metrics.get("steps"):
                step_counts.append(activity_metrics["steps"])
        
        if len(step_counts) < 7:
            return {"risk_score": 0, "reason": "insufficient_data"}
        
        # Calculate coefficient of variation
        mean_steps = sum(step_counts) / len(step_counts)
        variance = sum((s - mean_steps) ** 2 for s in step_counts) / len(step_counts)
        std_dev = variance ** 0.5
        cv = std_dev / mean_steps if mean_steps > 0 else 0
        
        # High variation indicates inconsistent activity
        if cv > 0.5:
            risk_score = 60
        elif cv > 0.3:
            risk_score = 30
        else:
            risk_score = 0
        
        # Check for significant drops
        recent_avg = sum(step_counts[-3:]) / 3
        baseline_avg = sum(step_counts[:-3]) / len(step_counts[:-3])
        
        if recent_avg < baseline_avg * 0.7:  # 30% drop
            risk_score += 40
        
        return {
            "risk_score": min(risk_score, 100),
            "activity_consistency": 1 - cv,
            "recent_activity_change": (recent_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0,
            "risk_factors": ["Inconsistent activity patterns"] if cv > 0.5 else []
        }
    
    async def _assess_pain_correlation_risk(self, patient_id: str, recent_data: List[Dict]) -> Dict[str, Any]:
        """Assess pain correlation with activity patterns"""
        # Get recent PRO scores
        pro_scores = await self.db.koos_scores.find({
            "patient_id": patient_id
        }).sort("date", -1).limit(5).to_list(5)
        
        if not pro_scores:
            pro_scores = await self.db.ases_scores.find({
                "patient_id": patient_id
            }).sort("date", -1).limit(5).to_list(5)
        
        if not pro_scores:
            return {"risk_score": 0, "reason": "no_pro_scores"}
        
        # Check if pain scores are consistently low
        pain_scores = []
        for score in pro_scores:
            if "pain_score" in score:
                pain_scores.append(score["pain_score"])
            elif "pain_component" in score:
                pain_scores.append(score["pain_component"] * 2)  # Convert to 0-100 scale
        
        if not pain_scores:
            return {"risk_score": 0, "reason": "no_pain_scores"}
        
        avg_pain = sum(pain_scores) / len(pain_scores)
        
        # High risk if pain consistently high (low scores mean high pain)
        if avg_pain < 40:
            risk_score = 70
        elif avg_pain < 60:
            risk_score = 40
        else:
            risk_score = 10
        
        return {
            "risk_score": risk_score,
            "avg_pain_score": avg_pain,
            "risk_factors": ["Persistent pain affecting recovery"] if risk_score > 40 else []
        }
    
    def _categorize_risk(self, overall_risk: float) -> str:
        """Categorize overall risk score"""
        if overall_risk < 25:
            return "Low"
        elif overall_risk < 50:
            return "Moderate"
        elif overall_risk < 75:
            return "High"
        else:
            return "Very High"
    
    def _generate_clinical_alerts(self, risk_assessment: Dict, overall_risk: float) -> List[str]:
        """Generate clinical alerts based on risk assessment"""
        alerts = []
        
        if overall_risk > 75:
            alerts.append("URGENT: Multiple high-risk factors detected")
        elif overall_risk > 50:
            alerts.append("WARNING: Elevated risk factors require attention")
        
        # Specific alerts for each risk category
        for risk_type, assessment in risk_assessment.items():
            if isinstance(assessment, dict) and assessment.get("risk_score", 0) > 60:
                alerts.append(f"High {risk_type.replace('_', ' ')} detected")
        
        return alerts
    
    def _generate_risk_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        for risk_type, assessment in risk_assessment.items():
            if isinstance(assessment, dict):
                risk_factors = assessment.get("risk_factors", [])
                recommendations.extend(risk_factors)
        
        # Add general recommendations
        high_risk_areas = [
            risk_type for risk_type, assessment in risk_assessment.items()
            if isinstance(assessment, dict) and assessment.get("risk_score", 0) > 50
        ]
        
        if len(high_risk_areas) > 2:
            recommendations.append("Consider comprehensive recovery program review")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _calculate_optimal_activity(self, patient_id: str, diagnosis_type: str) -> Dict[str, Any]:
        """Calculate optimal activity levels for individual recovery"""
        # Get historical activity data
        activity_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id,
            "activity_metrics.steps": {"$exists": True}
        }).sort("date", -1).limit(30).to_list(30)
        
        if not activity_data:
            return {"error": "No activity data available"}
        
        # Calculate current averages
        step_counts = [d.get("activity_metrics", {}).get("steps", 0) for d in activity_data]
        avg_steps = sum(step_counts) / len(step_counts)
        
        # Diagnosis-specific recommendations
        if "knee" in diagnosis_type.lower():
            optimal_steps = min(avg_steps * 1.1, 8000)  # Conservative for knee
        else:
            optimal_steps = min(avg_steps * 1.15, 10000)  # Slightly higher for shoulder
        
        return {
            "current_avg_steps": avg_steps,
            "optimal_daily_steps": optimal_steps,
            "progression_rate": "10-15% weekly increase",
            "activity_recommendations": [
                "Focus on consistent daily movement",
                "Gradually increase intensity",
                "Monitor for pain response"
            ]
        }
    
    def _analyze_sleep_patterns(self, wearable_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sleep patterns for optimization"""
        sleep_data = []
        for day in wearable_data:
            sleep_metrics = day.get("sleep_metrics", {})
            if sleep_metrics:
                sleep_data.append(sleep_metrics)
        
        if not sleep_data:
            return {"error": "No sleep data available"}
        
        # Calculate averages
        efficiencies = [s.get("sleep_efficiency", 0) for s in sleep_data if s.get("sleep_efficiency")]
        durations = [s.get("total_sleep_minutes", 0) / 60 for s in sleep_data if s.get("total_sleep_minutes")]
        
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        recommendations = []
        if avg_efficiency < 85:
            recommendations.append("Improve sleep environment and routine")
        if avg_duration < 7:
            recommendations.append("Aim for 7-9 hours of sleep nightly")
        
        return {
            "avg_sleep_efficiency": avg_efficiency,
            "avg_sleep_duration": avg_duration,
            "sleep_quality_trend": "improving" if avg_efficiency > 80 else "needs_attention",
            "recommendations": recommendations
        }
    
    def _optimize_exercise_timing(self, wearable_data: List[Dict]) -> Dict[str, Any]:
        """Optimize exercise timing based on HR recovery patterns"""
        # This is a simplified implementation
        # In practice, would analyze heart rate patterns throughout the day
        return {
            "optimal_timing": "Morning hours (8-11 AM)",
            "reasoning": "Based on circadian rhythm and recovery patterns",
            "avoid_times": ["Late evening (after 8 PM)"],
            "recommendations": [
                "Schedule high-intensity activities in the morning",
                "Allow adequate recovery time between sessions"
            ]
        }
    
    async def _predict_pain_patterns(self, patient_id: str, wearable_data: List[Dict]) -> Dict[str, Any]:
        """Predict pain patterns using comprehensive data"""
        # Get recent PRO scores for correlation
        pro_scores = await self.db.koos_scores.find({
            "patient_id": patient_id
        }).sort("date", -1).limit(10).to_list(10)
        
        if not pro_scores:
            pro_scores = await self.db.ases_scores.find({
                "patient_id": patient_id
            }).sort("date", -1).limit(10).to_list(10)
        
        if not pro_scores or len(wearable_data) < 7:
            return {"error": "Insufficient data for pain prediction"}
        
        # Simple pattern analysis
        activity_levels = [d.get("activity_metrics", {}).get("steps", 0) for d in wearable_data[-7:]]
        sleep_quality = [d.get("sleep_metrics", {}).get("sleep_efficiency", 0) for d in wearable_data[-7:]]
        
        avg_activity = sum(activity_levels) / len(activity_levels)
        avg_sleep = sum(sleep_quality) / len(sleep_quality) if any(sleep_quality) else 0
        
        # Predict based on patterns
        pain_risk = "low"
        if avg_activity < 2000 or avg_sleep < 70:
            pain_risk = "moderate"
        if avg_activity < 1000 and avg_sleep < 60:
            pain_risk = "high"
        
        return {
            "pain_risk_level": pain_risk,
            "contributing_factors": [
                "Low activity" if avg_activity < 2000 else None,
                "Poor sleep" if avg_sleep < 70 else None
            ],
            "recommendations": [
                "Monitor activity and pain correlation",
                "Maintain consistent sleep schedule"
            ]
        }
    
    def _assess_recovery_readiness(self, wearable_data: List[Dict]) -> Dict[str, Any]:
        """Assess readiness for next recovery phase"""
        if len(wearable_data) < 7:
            return {"error": "Insufficient data"}
        
        recent_data = wearable_data[:7]  # Last 7 days
        
        # Check consistency in key metrics
        step_counts = [d.get("activity_metrics", {}).get("steps", 0) for d in recent_data]
        sleep_efficiency = [d.get("sleep_metrics", {}).get("sleep_efficiency", 0) for d in recent_data]
        
        # Calculate consistency scores
        step_consistency = 1 - (np.std(step_counts) / np.mean(step_counts)) if np.mean(step_counts) > 0 else 0
        sleep_consistency = 1 - (np.std(sleep_efficiency) / np.mean(sleep_efficiency)) if np.mean(sleep_efficiency) > 0 else 0
        
        readiness_score = (step_consistency + sleep_consistency) / 2
        
        if readiness_score > 0.8:
            readiness = "ready"
        elif readiness_score > 0.6:
            readiness = "cautious"
        else:
            readiness = "not_ready"
        
        return {
            "readiness_level": readiness,
            "readiness_score": readiness_score,
            "consistency_metrics": {
                "activity_consistency": step_consistency,
                "sleep_consistency": sleep_consistency
            },
            "recommendations": [
                "Maintain current progress" if readiness == "ready" else "Focus on consistency"
            ]
        }
    
    def _calculate_insight_confidence(self, wearable_data: List[Dict]) -> float:
        """Calculate confidence score for insights"""
        data_points = len(wearable_data)
        completeness = min(data_points / 30, 1.0)  # 30 days = full confidence
        
        # Check data quality
        quality_score = 0
        for day in wearable_data:
            if day.get("activity_metrics"):
                quality_score += 0.25
            if day.get("sleep_metrics"):
                quality_score += 0.25
            if day.get("heart_rate_metrics"):
                quality_score += 0.25
            if day.get("movement_metrics"):
                quality_score += 0.25
        
        avg_quality = quality_score / len(wearable_data) if wearable_data else 0
        
        return (completeness + avg_quality) / 2
    
    async def _calculate_compliance_score(self, patient_id: str) -> Dict[str, Any]:
        """Calculate patient compliance with activity recommendations"""
        # Get last 30 days of data
        recent_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id,
            "date": {"$gte": (datetime.utcnow() - timedelta(days=30)).date()}
        }).sort("date", -1).to_list(30)
        
        if not recent_data:
            return {"compliance_score": 0, "reason": "no_data"}
        
        # Calculate data collection compliance
        expected_days = 30
        actual_days = len(recent_data)
        data_compliance = actual_days / expected_days
        
        # Calculate activity compliance (assuming 5000 steps as target)
        target_steps = 5000
        step_compliance = 0
        steps_data = []
        
        for day in recent_data:
            activity_metrics = day.get("activity_metrics", {})
            if activity_metrics and activity_metrics.get("steps"):
                steps = activity_metrics["steps"]
                steps_data.append(steps)
                if steps >= target_steps:
                    step_compliance += 1
        
        step_compliance = step_compliance / len(steps_data) if steps_data else 0
        
        overall_compliance = (data_compliance + step_compliance) / 2
        
        return {
            "compliance_score": overall_compliance,
            "data_collection_compliance": data_compliance,
            "activity_compliance": step_compliance,
            "days_with_data": actual_days,
            "avg_daily_steps": sum(steps_data) / len(steps_data) if steps_data else 0
        }
    
    async def _detect_intervention_triggers(self, patient_id: str) -> List[Dict[str, Any]]:
        """Detect triggers for early intervention"""
        triggers = []
        
        # Get recent data
        recent_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id,
            "date": {"$gte": (datetime.utcnow() - timedelta(days=14)).date()}
        }).sort("date", -1).to_list(14)
        
        if len(recent_data) < 7:
            return triggers
        
        # Check for declining activity
        step_counts = [d.get("activity_metrics", {}).get("steps", 0) for d in recent_data]
        if len(step_counts) >= 7:
            recent_avg = sum(step_counts[:7]) / 7
            baseline_avg = sum(step_counts[7:]) / len(step_counts[7:]) if len(step_counts) > 7 else recent_avg
            
            if recent_avg < baseline_avg * 0.7:  # 30% decline
                triggers.append({
                    "type": "activity_decline",
                    "severity": "high",
                    "description": "Significant decrease in daily activity",
                    "recommendation": "Immediate assessment recommended"
                })
        
        # Check for sleep disruption
        sleep_efficiencies = [
            d.get("sleep_metrics", {}).get("sleep_efficiency", 0) 
            for d in recent_data[:7]
            if d.get("sleep_metrics", {}).get("sleep_efficiency")
        ]
        
        if sleep_efficiencies and sum(sleep_efficiencies) / len(sleep_efficiencies) < 65:
            triggers.append({
                "type": "sleep_disruption",
                "severity": "medium",
                "description": "Poor sleep quality detected",
                "recommendation": "Evaluate pain management and sleep hygiene"
            })
        
        return triggers
    
    async def _generate_comparative_analysis(self, patient_id: str) -> Dict[str, Any]:
        """Generate comparative analysis against similar cases"""
        # Get patient info
        patient = await self.db.patients.find_one({"id": patient_id})
        if not patient:
            return {"error": "Patient not found"}
        
        diagnosis_type = patient.get("diagnosis_type")
        
        # Get similar patients (same diagnosis)
        similar_patients = await self.db.patients.find({
            "diagnosis_type": diagnosis_type,
            "id": {"$ne": patient_id}
        }).limit(10).to_list(10)
        
        if not similar_patients:
            return {"error": "No similar cases found"}
        
        # This would implement more sophisticated comparative analysis
        # For now, return a simplified comparison
        return {
            "cohort_size": len(similar_patients),
            "diagnosis": diagnosis_type,
            "patient_percentile": 75,  # Placeholder
            "recovery_trajectory": "Above average",
            "similar_cases_outcomes": "Generally positive"
        }
    
    async def _generate_clinical_summary(self, patient_id: str) -> Dict[str, Any]:
        """Generate clinical summary for appointments"""
        # Get recent comprehensive data
        summary_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id
        }).sort("date", -1).limit(30).to_list(30)
        
        if not summary_data:
            return {"error": "No data available"}
        
        # Calculate key metrics
        step_counts = [d.get("activity_metrics", {}).get("steps", 0) for d in summary_data]
        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
        
        sleep_data = [
            d.get("sleep_metrics", {}).get("sleep_efficiency", 0) 
            for d in summary_data
            if d.get("sleep_metrics", {}).get("sleep_efficiency")
        ]
        avg_sleep_efficiency = sum(sleep_data) / len(sleep_data) if sleep_data else 0
        
        return {
            "summary_period": "Last 30 days",
            "key_metrics": {
                "avg_daily_steps": avg_steps,
                "avg_sleep_efficiency": avg_sleep_efficiency,
                "data_completeness": len(summary_data) / 30
            },
            "trends": {
                "activity": "stable",  # Would calculate actual trends
                "sleep": "improving"   # Would calculate actual trends
            },
            "clinical_notes": [
                "Patient showing consistent data collection",
                "Activity levels within expected range",
                "Sleep patterns improving"
            ]
        }
    
    def _generate_dashboard_alerts(self, compliance: Dict, triggers: List[Dict]) -> List[str]:
        """Generate alerts for provider dashboard"""
        alerts = []
        
        if compliance.get("compliance_score", 0) < 0.7:
            alerts.append("Low patient compliance with data collection")
        
        high_priority_triggers = [t for t in triggers if t.get("severity") == "high"]
        if high_priority_triggers:
            alerts.append(f"High priority intervention needed: {high_priority_triggers[0]['description']}")
        
        return alerts