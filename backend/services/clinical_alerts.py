"""
Clinical alerts and recommendation engine for orthopedic recovery tracking.
Generates real-time alerts and evidence-based recommendations.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from enum import Enum
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertCategory(str, Enum):
    """Alert categories"""
    PAIN_MANAGEMENT = "pain_management"
    ACTIVITY_DECLINE = "activity_decline"
    SLEEP_DISRUPTION = "sleep_disruption"
    CARDIOVASCULAR = "cardiovascular"
    MOBILITY_CONCERN = "mobility_concern"
    COMPLIANCE = "compliance"
    PLATEAU = "plateau"
    INFECTION_RISK = "infection_risk"
    PSYCHOLOGICAL = "psychological"


class RecommendationLevel(str, Enum):
    """Recommendation priority levels"""
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    ROUTINE = "routine"
    PREVENTIVE = "preventive"


@dataclass
class ClinicalAlert:
    """Clinical alert data structure"""
    alert_id: str
    patient_id: str
    alert_type: AlertCategory
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    triggered_by: str  # Metric or condition that triggered alert
    trigger_value: Optional[float]
    threshold_value: Optional[float]
    recommendations: List[str]
    evidence_level: str  # "high", "medium", "low"
    requires_immediate_attention: bool
    auto_resolve: bool
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]


@dataclass
class ClinicalRecommendation:
    """Clinical recommendation data structure"""
    recommendation_id: str
    patient_id: str
    category: AlertCategory
    priority: RecommendationLevel
    title: str
    description: str
    rationale: str
    evidence_level: str
    action_items: List[str]
    target_metric: Optional[str]
    target_improvement: Optional[float]
    timeline: str
    contraindications: List[str]
    generated_at: datetime
    implemented: bool
    implementation_date: Optional[datetime]
    effectiveness_score: Optional[float]


class ClinicalAlertsEngine:
    """Engine for generating clinical alerts and recommendations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.recommendation_templates = self._initialize_recommendation_templates()
    
    async def generate_real_time_alerts(self, patient_id: str) -> List[ClinicalAlert]:
        """
        Generate real-time clinical alerts based on current patient data
        """
        try:
            alerts = []
            
            # Get patient information
            patient = await self.db.patients.find_one({"id": patient_id})
            if not patient:
                return alerts
            
            # Get recent data (last 7 days)
            recent_data = await self._get_recent_patient_data(patient_id, days=7)
            
            if not recent_data:
                return alerts
            
            # Check for various alert conditions
            alerts.extend(await self._check_pain_alerts(patient_id, recent_data))
            alerts.extend(await self._check_activity_alerts(patient_id, recent_data))
            alerts.extend(await self._check_sleep_alerts(patient_id, recent_data))
            alerts.extend(await self._check_cardiovascular_alerts(patient_id, recent_data))
            alerts.extend(await self._check_mobility_alerts(patient_id, recent_data))
            alerts.extend(await self._check_compliance_alerts(patient_id, recent_data))
            alerts.extend(await self._check_plateau_alerts(patient_id, recent_data))
            alerts.extend(await self._check_infection_risk_alerts(patient_id, recent_data))
            
            # Store alerts in database
            for alert in alerts:
                await self._store_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Real-time alert generation failed for {patient_id}: {e}")
            return []
    
    async def generate_clinical_recommendations(self, patient_id: str) -> List[ClinicalRecommendation]:
        """
        Generate evidence-based clinical recommendations
        """
        try:
            recommendations = []
            
            # Get patient data
            patient = await self.db.patients.find_one({"id": patient_id})
            if not patient:
                return recommendations
            
            # Get comprehensive patient data
            patient_data = await self._get_comprehensive_patient_data(patient_id)
            
            # Generate recommendations based on current state
            recommendations.extend(await self._generate_pain_recommendations(patient_id, patient_data))
            recommendations.extend(await self._generate_activity_recommendations(patient_id, patient_data))
            recommendations.extend(await self._generate_sleep_recommendations(patient_id, patient_data))
            recommendations.extend(await self._generate_mobility_recommendations(patient_id, patient_data))
            recommendations.extend(await self._generate_recovery_optimization_recommendations(patient_id, patient_data))
            
            # Store recommendations
            for recommendation in recommendations:
                await self._store_recommendation(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Clinical recommendation generation failed for {patient_id}: {e}")
            return []
    
    async def assess_intervention_triggers(self, patient_id: str) -> Dict[str, Any]:
        """
        Assess whether clinical intervention is needed
        """
        try:
            # Get active alerts
            active_alerts = await self._get_active_alerts(patient_id)
            
            # Calculate intervention urgency
            intervention_score = self._calculate_intervention_score(active_alerts)
            
            # Determine intervention level
            intervention_level = self._determine_intervention_level(intervention_score)
            
            # Generate intervention recommendations
            interventions = await self._generate_intervention_recommendations(patient_id, active_alerts, intervention_level)
            
            return {
                "patient_id": patient_id,
                "intervention_score": intervention_score,
                "intervention_level": intervention_level,
                "active_alerts": len(active_alerts),
                "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "recommended_interventions": interventions,
                "requires_immediate_attention": intervention_score >= 80,
                "assessment_timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Intervention assessment failed for {patient_id}: {e}")
            return {"error": str(e)}
    
    async def generate_provider_notifications(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Generate notifications for healthcare providers
        """
        try:
            notifications = []
            
            # Get critical alerts
            critical_alerts = await self._get_alerts_by_severity(patient_id, AlertSeverity.CRITICAL)
            
            for alert in critical_alerts:
                notifications.append({
                    "type": "critical_alert",
                    "patient_id": patient_id,
                    "alert_category": alert.alert_type,
                    "message": alert.title,
                    "description": alert.description,
                    "urgency": "immediate",
                    "triggered_at": alert.triggered_at,
                    "recommendations": alert.recommendations
                })
            
            # Check for missed appointments or assessments
            missed_assessments = await self._check_missed_assessments(patient_id)
            if missed_assessments:
                notifications.append({
                    "type": "missed_assessment",
                    "patient_id": patient_id,
                    "message": "Patient has missed scheduled assessments",
                    "description": f"Missed {missed_assessments} PRO assessments",
                    "urgency": "routine",
                    "recommendations": ["Schedule follow-up assessment", "Contact patient for compliance check"]
                })
            
            # Check for declining trends
            declining_trends = await self._check_declining_trends(patient_id)
            if declining_trends:
                notifications.append({
                    "type": "declining_trend",
                    "patient_id": patient_id,
                    "message": "Declining recovery metrics detected",
                    "description": f"Declining trends in: {', '.join(declining_trends)}",
                    "urgency": "routine",
                    "recommendations": ["Review current treatment plan", "Consider additional interventions"]
                })
            
            return notifications
            
        except Exception as e:
            logger.error(f"Provider notification generation failed for {patient_id}: {e}")
            return []
    
    async def _get_recent_patient_data(self, patient_id: str, days: int = 7) -> Dict[str, Any]:
        """Get recent patient data for alert generation"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)
        
        # Get wearable data
        wearable_data = await self.db.comprehensive_wearable_data.find({
            "patient_id": patient_id,
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", -1).to_list(days)
        
        # Get PRO scores
        patient = await self.db.patients.find_one({"id": patient_id})
        diagnosis_type = patient.get("diagnosis_type", "") if patient else ""
        
        knee_diagnoses = ["ACL Tear", "Meniscus Tear", "Cartilage Defect", "Knee Osteoarthritis", "Post Total Knee Replacement"]
        is_knee = diagnosis_type in knee_diagnoses
        
        if is_knee:
            pro_scores = await self.db.koos_scores.find({
                "patient_id": patient_id,
                "date": {"$gte": start_date, "$lte": end_date}
            }).sort("date", -1).to_list(10)
        else:
            pro_scores = await self.db.ases_scores.find({
                "patient_id": patient_id,
                "date": {"$gte": start_date, "$lte": end_date}
            }).sort("date", -1).to_list(10)
        
        return {
            "wearable_data": wearable_data,
            "pro_scores": pro_scores,
            "patient_info": patient,
            "is_knee_patient": is_knee,
            "days_analyzed": days
        }
    
    async def _check_pain_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for pain-related alerts"""
        alerts = []
        
        pro_scores = recent_data.get("pro_scores", [])
        if not pro_scores:
            return alerts
        
        latest_score = pro_scores[0]
        is_knee = recent_data.get("is_knee_patient", True)
        
        if is_knee:
            pain_score = latest_score.get("pain_score", 0)
            # KOOS: higher scores = less pain
            if pain_score < 30:  # Severe pain
                alerts.append(ClinicalAlert(
                    alert_id=f"pain_severe_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.PAIN_MANAGEMENT,
                    severity=AlertSeverity.CRITICAL,
                    title="Severe Pain Detected",
                    description=f"KOOS pain score of {pain_score:.1f} indicates severe pain",
                    triggered_at=datetime.utcnow(),
                    triggered_by="pain_score",
                    trigger_value=pain_score,
                    threshold_value=30,
                    recommendations=[
                        "Review pain management protocol",
                        "Consider medication adjustment",
                        "Evaluate for complications"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
            elif pain_score < 50:  # Moderate pain
                alerts.append(ClinicalAlert(
                    alert_id=f"pain_moderate_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.PAIN_MANAGEMENT,
                    severity=AlertSeverity.MEDIUM,
                    title="Elevated Pain Levels",
                    description=f"KOOS pain score of {pain_score:.1f} indicates elevated pain",
                    triggered_at=datetime.utcnow(),
                    triggered_by="pain_score",
                    trigger_value=pain_score,
                    threshold_value=50,
                    recommendations=[
                        "Monitor pain trends",
                        "Consider pain management optimization",
                        "Assess activity modifications"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        else:
            # ASES pain component
            pain_component = latest_score.get("pain_component", 0)
            # ASES: higher scores = less pain (0-50 scale)
            if pain_component < 15:  # Severe pain
                alerts.append(ClinicalAlert(
                    alert_id=f"pain_severe_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.PAIN_MANAGEMENT,
                    severity=AlertSeverity.CRITICAL,
                    title="Severe Shoulder Pain Detected",
                    description=f"ASES pain component of {pain_component:.1f} indicates severe pain",
                    triggered_at=datetime.utcnow(),
                    triggered_by="pain_component",
                    trigger_value=pain_component,
                    threshold_value=15,
                    recommendations=[
                        "Immediate pain assessment",
                        "Review shoulder positioning",
                        "Consider imaging if indicated"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_activity_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for activity-related alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        if len(wearable_data) < 3:
            return alerts
        
        # Check for significant activity decline
        recent_steps = []
        for day in wearable_data[:3]:  # Last 3 days
            activity = day.get("activity_metrics", {})
            if activity and activity.get("steps"):
                recent_steps.append(activity["steps"])
        
        baseline_steps = []
        for day in wearable_data[3:]:  # Previous days
            activity = day.get("activity_metrics", {})
            if activity and activity.get("steps"):
                baseline_steps.append(activity["steps"])
        
        if recent_steps and baseline_steps:
            recent_avg = sum(recent_steps) / len(recent_steps)
            baseline_avg = sum(baseline_steps) / len(baseline_steps)
            
            decline_percent = (baseline_avg - recent_avg) / baseline_avg * 100
            
            if decline_percent > 50:  # >50% decline
                alerts.append(ClinicalAlert(
                    alert_id=f"activity_decline_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.ACTIVITY_DECLINE,
                    severity=AlertSeverity.HIGH,
                    title="Significant Activity Decline",
                    description=f"Activity decreased by {decline_percent:.1f}% (from {baseline_avg:.0f} to {recent_avg:.0f} steps)",
                    triggered_at=datetime.utcnow(),
                    triggered_by="daily_steps",
                    trigger_value=recent_avg,
                    threshold_value=baseline_avg * 0.5,
                    recommendations=[
                        "Assess for pain or discomfort",
                        "Review activity restrictions",
                        "Consider physical therapy consultation"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
            elif decline_percent > 30:  # >30% decline
                alerts.append(ClinicalAlert(
                    alert_id=f"activity_decline_moderate_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.ACTIVITY_DECLINE,
                    severity=AlertSeverity.MEDIUM,
                    title="Moderate Activity Decline",
                    description=f"Activity decreased by {decline_percent:.1f}%",
                    triggered_at=datetime.utcnow(),
                    triggered_by="daily_steps",
                    trigger_value=recent_avg,
                    threshold_value=baseline_avg * 0.7,
                    recommendations=[
                        "Monitor activity trends",
                        "Encourage gradual activity increase",
                        "Check patient motivation"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        # Check for very low absolute activity
        if recent_steps:
            avg_recent_steps = sum(recent_steps) / len(recent_steps)
            if avg_recent_steps < 1000:  # Very low activity
                alerts.append(ClinicalAlert(
                    alert_id=f"activity_low_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.ACTIVITY_DECLINE,
                    severity=AlertSeverity.HIGH,
                    title="Very Low Activity Level",
                    description=f"Average daily steps: {avg_recent_steps:.0f}",
                    triggered_at=datetime.utcnow(),
                    triggered_by="daily_steps",
                    trigger_value=avg_recent_steps,
                    threshold_value=1000,
                    recommendations=[
                        "Assess mobility limitations",
                        "Review weight-bearing restrictions",
                        "Consider mobility aids if appropriate"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_sleep_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for sleep-related alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        if not wearable_data:
            return alerts
        
        # Collect sleep metrics
        sleep_efficiencies = []
        sleep_durations = []
        
        for day in wearable_data:
            sleep_metrics = day.get("sleep_metrics", {})
            if sleep_metrics:
                if sleep_metrics.get("sleep_efficiency"):
                    sleep_efficiencies.append(sleep_metrics["sleep_efficiency"])
                if sleep_metrics.get("total_sleep_minutes"):
                    sleep_durations.append(sleep_metrics["total_sleep_minutes"] / 60)
        
        # Check sleep efficiency
        if sleep_efficiencies:
            avg_efficiency = sum(sleep_efficiencies) / len(sleep_efficiencies)
            if avg_efficiency < 60:  # Poor sleep efficiency
                alerts.append(ClinicalAlert(
                    alert_id=f"sleep_poor_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.SLEEP_DISRUPTION,
                    severity=AlertSeverity.MEDIUM,
                    title="Poor Sleep Efficiency",
                    description=f"Average sleep efficiency: {avg_efficiency:.1f}%",
                    triggered_at=datetime.utcnow(),
                    triggered_by="sleep_efficiency",
                    trigger_value=avg_efficiency,
                    threshold_value=60,
                    recommendations=[
                        "Assess sleep environment",
                        "Review pain management for nighttime",
                        "Consider sleep hygiene counseling"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        # Check sleep duration
        if sleep_durations:
            avg_duration = sum(sleep_durations) / len(sleep_durations)
            if avg_duration < 5:  # Very short sleep
                alerts.append(ClinicalAlert(
                    alert_id=f"sleep_short_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.SLEEP_DISRUPTION,
                    severity=AlertSeverity.HIGH,
                    title="Insufficient Sleep Duration",
                    description=f"Average sleep duration: {avg_duration:.1f} hours",
                    triggered_at=datetime.utcnow(),
                    triggered_by="sleep_duration",
                    trigger_value=avg_duration,
                    threshold_value=5,
                    recommendations=[
                        "Assess for sleep disorders",
                        "Review medications affecting sleep",
                        "Consider sleep study if indicated"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=False,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_cardiovascular_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for cardiovascular-related alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        if not wearable_data:
            return alerts
        
        # Collect heart rate data
        resting_hrs = []
        
        for day in wearable_data:
            hr_metrics = day.get("heart_rate_metrics", {})
            if hr_metrics and hr_metrics.get("resting_hr"):
                resting_hrs.append(hr_metrics["resting_hr"])
        
        if resting_hrs:
            avg_resting_hr = sum(resting_hrs) / len(resting_hrs)
            
            # Check for elevated resting heart rate
            if avg_resting_hr > 100:  # Tachycardia
                alerts.append(ClinicalAlert(
                    alert_id=f"hr_elevated_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.CARDIOVASCULAR,
                    severity=AlertSeverity.HIGH,
                    title="Elevated Resting Heart Rate",
                    description=f"Average resting HR: {avg_resting_hr:.0f} bpm",
                    triggered_at=datetime.utcnow(),
                    triggered_by="resting_heart_rate",
                    trigger_value=avg_resting_hr,
                    threshold_value=100,
                    recommendations=[
                        "Assess for fever or infection",
                        "Review medications",
                        "Consider cardiovascular evaluation"
                    ],
                    evidence_level="high",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
            elif avg_resting_hr > 90:  # Moderately elevated
                alerts.append(ClinicalAlert(
                    alert_id=f"hr_moderate_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.CARDIOVASCULAR,
                    severity=AlertSeverity.MEDIUM,
                    title="Moderately Elevated Heart Rate",
                    description=f"Average resting HR: {avg_resting_hr:.0f} bpm",
                    triggered_at=datetime.utcnow(),
                    triggered_by="resting_heart_rate",
                    trigger_value=avg_resting_hr,
                    threshold_value=90,
                    recommendations=[
                        "Monitor heart rate trends",
                        "Assess stress levels",
                        "Ensure adequate hydration"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_mobility_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for mobility-related alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        if not wearable_data:
            return alerts
        
        # Collect walking speed data
        walking_speeds = []
        
        for day in wearable_data:
            movement_metrics = day.get("movement_metrics", {})
            if movement_metrics and movement_metrics.get("walking_speed_ms"):
                walking_speeds.append(movement_metrics["walking_speed_ms"])
        
        if walking_speeds:
            avg_walking_speed = sum(walking_speeds) / len(walking_speeds)
            
            # Check for very slow walking
            if avg_walking_speed < 0.8:  # Very slow walking
                alerts.append(ClinicalAlert(
                    alert_id=f"mobility_slow_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.MOBILITY_CONCERN,
                    severity=AlertSeverity.MEDIUM,
                    title="Reduced Walking Speed",
                    description=f"Average walking speed: {avg_walking_speed:.2f} m/s",
                    triggered_at=datetime.utcnow(),
                    triggered_by="walking_speed",
                    trigger_value=avg_walking_speed,
                    threshold_value=0.8,
                    recommendations=[
                        "Assess gait pattern",
                        "Review weight-bearing status",
                        "Consider gait training"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_compliance_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for compliance-related alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        days_analyzed = recent_data.get("days_analyzed", 7)
        
        # Check data collection compliance
        data_compliance = len(wearable_data) / days_analyzed
        
        if data_compliance < 0.5:  # Less than 50% data collection
            alerts.append(ClinicalAlert(
                alert_id=f"compliance_data_{patient_id}_{datetime.utcnow().isoformat()}",
                patient_id=patient_id,
                alert_type=AlertCategory.COMPLIANCE,
                severity=AlertSeverity.MEDIUM,
                title="Poor Data Collection Compliance",
                description=f"Only {data_compliance*100:.0f}% data collection compliance",
                triggered_at=datetime.utcnow(),
                triggered_by="data_collection",
                trigger_value=data_compliance,
                threshold_value=0.5,
                recommendations=[
                    "Contact patient about device usage",
                    "Assess technical issues",
                    "Provide additional training if needed"
                ],
                evidence_level="high",
                requires_immediate_attention=False,
                auto_resolve=True,
                resolved_at=None,
                resolved_by=None
            ))
        
        return alerts
    
    async def _check_plateau_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for recovery plateau alerts"""
        alerts = []
        
        pro_scores = recent_data.get("pro_scores", [])
        if len(pro_scores) < 3:
            return alerts
        
        # Check for plateau in PRO scores
        total_scores = [score.get("total_score", 0) for score in pro_scores[:3]]
        
        if len(total_scores) >= 3:
            # Check if scores are stagnant (little change over time)
            score_range = max(total_scores) - min(total_scores)
            if score_range < 5 and max(total_scores) < 80:  # Stagnant and not at high level
                alerts.append(ClinicalAlert(
                    alert_id=f"plateau_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.PLATEAU,
                    severity=AlertSeverity.MEDIUM,
                    title="Recovery Plateau Detected",
                    description=f"PRO scores showing minimal change (range: {score_range:.1f})",
                    triggered_at=datetime.utcnow(),
                    triggered_by="pro_scores",
                    trigger_value=score_range,
                    threshold_value=5,
                    recommendations=[
                        "Review current treatment plan",
                        "Consider therapy progression",
                        "Assess patient motivation"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=False,
                    auto_resolve=True,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    async def _check_infection_risk_alerts(self, patient_id: str, recent_data: Dict) -> List[ClinicalAlert]:
        """Check for infection risk alerts"""
        alerts = []
        
        wearable_data = recent_data.get("wearable_data", [])
        if not wearable_data:
            return alerts
        
        # Look for patterns suggesting infection (elevated HR + reduced activity)
        recent_hr = []
        recent_activity = []
        
        for day in wearable_data[:3]:  # Last 3 days
            hr_metrics = day.get("heart_rate_metrics", {})
            activity_metrics = day.get("activity_metrics", {})
            
            if hr_metrics and hr_metrics.get("resting_hr"):
                recent_hr.append(hr_metrics["resting_hr"])
            
            if activity_metrics and activity_metrics.get("steps"):
                recent_activity.append(activity_metrics["steps"])
        
        if recent_hr and recent_activity:
            avg_hr = sum(recent_hr) / len(recent_hr)
            avg_activity = sum(recent_activity) / len(recent_activity)
            
            # Elevated HR + very low activity might suggest infection
            if avg_hr > 95 and avg_activity < 1500:
                alerts.append(ClinicalAlert(
                    alert_id=f"infection_risk_{patient_id}_{datetime.utcnow().isoformat()}",
                    patient_id=patient_id,
                    alert_type=AlertCategory.INFECTION_RISK,
                    severity=AlertSeverity.HIGH,
                    title="Possible Infection Risk Pattern",
                    description=f"Elevated HR ({avg_hr:.0f} bpm) with low activity ({avg_activity:.0f} steps)",
                    triggered_at=datetime.utcnow(),
                    triggered_by="hr_activity_pattern",
                    trigger_value=avg_hr,
                    threshold_value=95,
                    recommendations=[
                        "Assess for signs of infection",
                        "Check surgical site if applicable",
                        "Consider fever evaluation"
                    ],
                    evidence_level="medium",
                    requires_immediate_attention=True,
                    auto_resolve=False,
                    resolved_at=None,
                    resolved_by=None
                ))
        
        return alerts
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict]:
        """Initialize alert thresholds for different metrics"""
        return {
            "pain_scores": {
                "koos_critical": 30,
                "koos_moderate": 50,
                "ases_critical": 15,
                "ases_moderate": 25
            },
            "activity": {
                "very_low_steps": 1000,
                "low_steps": 2000,
                "decline_threshold": 0.3
            },
            "sleep": {
                "poor_efficiency": 60,
                "very_poor_efficiency": 45,
                "short_duration": 5,
                "very_short_duration": 4
            },
            "heart_rate": {
                "tachycardia": 100,
                "elevated": 90,
                "very_elevated": 110
            },
            "walking_speed": {
                "very_slow": 0.8,
                "slow": 1.0
            }
        }
    
    def _initialize_recommendation_templates(self) -> Dict[str, Dict]:
        """Initialize recommendation templates"""
        return {
            "pain_management": {
                "severe": [
                    "Immediate pain assessment required",
                    "Review current pain medication regimen",
                    "Consider multimodal pain management approach",
                    "Evaluate for potential complications"
                ],
                "moderate": [
                    "Monitor pain trends closely",
                    "Optimize current pain management",
                    "Consider non-pharmacological interventions",
                    "Assess activity modifications"
                ]
            },
            "activity_optimization": {
                "low_activity": [
                    "Gradual activity progression protocol",
                    "Assess barriers to activity",
                    "Consider physical therapy referral",
                    "Monitor for overuse vs underuse"
                ],
                "declining_activity": [
                    "Identify cause of activity decline",
                    "Review weight-bearing restrictions",
                    "Assess pain-activity relationship",
                    "Consider motivation counseling"
                ]
            }
        }
    
    # Additional methods for storing alerts, calculating scores, etc.
    async def _store_alert(self, alert: ClinicalAlert) -> bool:
        """Store alert in database"""
        try:
            alert_data = {
                "alert_id": alert.alert_id,
                "patient_id": alert.patient_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "triggered_at": alert.triggered_at,
                "triggered_by": alert.triggered_by,
                "trigger_value": alert.trigger_value,
                "threshold_value": alert.threshold_value,
                "recommendations": alert.recommendations,
                "evidence_level": alert.evidence_level,
                "requires_immediate_attention": alert.requires_immediate_attention,
                "auto_resolve": alert.auto_resolve,
                "resolved_at": alert.resolved_at,
                "resolved_by": alert.resolved_by
            }
            
            await self.db.clinical_alerts.insert_one(alert_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            return False
    
    async def _store_recommendation(self, recommendation: ClinicalRecommendation) -> bool:
        """Store recommendation in database"""
        try:
            rec_data = {
                "recommendation_id": recommendation.recommendation_id,
                "patient_id": recommendation.patient_id,
                "category": recommendation.category.value,
                "priority": recommendation.priority.value,
                "title": recommendation.title,
                "description": recommendation.description,
                "rationale": recommendation.rationale,
                "evidence_level": recommendation.evidence_level,
                "action_items": recommendation.action_items,
                "target_metric": recommendation.target_metric,
                "target_improvement": recommendation.target_improvement,
                "timeline": recommendation.timeline,
                "contraindications": recommendation.contraindications,
                "generated_at": recommendation.generated_at,
                "implemented": recommendation.implemented,
                "implementation_date": recommendation.implementation_date,
                "effectiveness_score": recommendation.effectiveness_score
            }
            
            await self.db.clinical_recommendations.insert_one(rec_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")
            return False
    
    # Placeholder methods for comprehensive recommendation generation
    async def _get_comprehensive_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient data for recommendations"""
        # Implementation would gather all relevant patient data
        return {}
    
    async def _generate_pain_recommendations(self, patient_id: str, patient_data: Dict) -> List[ClinicalRecommendation]:
        """Generate pain management recommendations"""
        return []
    
    async def _generate_activity_recommendations(self, patient_id: str, patient_data: Dict) -> List[ClinicalRecommendation]:
        """Generate activity optimization recommendations"""
        return []
    
    async def _generate_sleep_recommendations(self, patient_id: str, patient_data: Dict) -> List[ClinicalRecommendation]:
        """Generate sleep optimization recommendations"""
        return []
    
    async def _generate_mobility_recommendations(self, patient_id: str, patient_data: Dict) -> List[ClinicalRecommendation]:
        """Generate mobility improvement recommendations"""
        return []
    
    async def _generate_recovery_optimization_recommendations(self, patient_id: str, patient_data: Dict) -> List[ClinicalRecommendation]:
        """Generate overall recovery optimization recommendations"""
        return []
    
    async def _get_active_alerts(self, patient_id: str) -> List[ClinicalAlert]:
        """Get active alerts for patient"""
        # Implementation would query database for unresolved alerts
        return []
    
    def _calculate_intervention_score(self, alerts: List[ClinicalAlert]) -> float:
        """Calculate intervention urgency score"""
        if not alerts:
            return 0.0
        
        score = 0
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                score += 30
            elif alert.severity == AlertSeverity.HIGH:
                score += 20
            elif alert.severity == AlertSeverity.MEDIUM:
                score += 10
            elif alert.severity == AlertSeverity.LOW:
                score += 5
        
        return min(score, 100)
    
    def _determine_intervention_level(self, score: float) -> str:
        """Determine intervention level based on score"""
        if score >= 80:
            return "immediate"
        elif score >= 60:
            return "urgent"
        elif score >= 40:
            return "routine"
        else:
            return "monitoring"
    
    async def _generate_intervention_recommendations(self, patient_id: str, alerts: List[ClinicalAlert], level: str) -> List[str]:
        """Generate intervention recommendations"""
        recommendations = []
        
        if level == "immediate":
            recommendations.extend([
                "Contact patient immediately",
                "Schedule urgent evaluation",
                "Review current treatment plan"
            ])
        elif level == "urgent":
            recommendations.extend([
                "Schedule evaluation within 24-48 hours",
                "Monitor closely",
                "Consider treatment modifications"
            ])
        elif level == "routine":
            recommendations.extend([
                "Schedule routine follow-up",
                "Continue monitoring",
                "Consider minor adjustments"
            ])
        
        return recommendations
    
    async def _get_alerts_by_severity(self, patient_id: str, severity: AlertSeverity) -> List[ClinicalAlert]:
        """Get alerts by severity level"""
        # Implementation would query database
        return []
    
    async def _check_missed_assessments(self, patient_id: str) -> int:
        """Check for missed PRO assessments"""
        # Implementation would check assessment schedule
        return 0
    
    async def _check_declining_trends(self, patient_id: str) -> List[str]:
        """Check for declining metric trends"""
        # Implementation would analyze trends
        return []