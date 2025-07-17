"""
Real-time Quality Monitoring System
Provides live data stream validation and quality tracking for clinical alerts
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import json
from motor.motor_asyncio import AsyncIOMotorClient

from ..models.wearable_data import WearableData
from ..services.data_validation import ClinicalDataValidator, DataQualityReport
from ..services.ml_data_quality import MLDataQualityAssurance


class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class QualityAlert:
    """Real-time quality alert"""
    alert_id: str
    patient_id: str
    timestamp: datetime
    priority: AlertPriority
    category: str
    message: str
    data_point: Dict[str, Any]
    quality_score: float
    recommended_action: str
    clinical_impact: str
    auto_resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class QualityMetrics:
    """Real-time quality metrics"""
    patient_id: str
    timestamp: datetime
    overall_quality_score: float
    completeness_score: float
    consistency_score: float
    reliability_score: float
    clinical_validity_score: float
    ml_readiness_score: float
    anomaly_score: float
    trend_score: float
    alert_count: int
    data_points_processed: int


@dataclass
class MonitoringConfiguration:
    """Configuration for quality monitoring"""
    patient_id: str
    monitoring_interval: int = 300  # seconds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.5,
        'high': 0.65,
        'medium': 0.8,
        'low': 0.9
    })
    alert_cooldown: int = 3600  # seconds
    anomaly_sensitivity: float = 0.8
    enable_ml_monitoring: bool = True
    enable_clinical_monitoring: bool = True
    notification_endpoints: List[str] = field(default_factory=list)


class RealTimeQualityMonitor:
    """
    Real-time quality monitoring system for continuous data validation
    Supports live data stream validation and clinical alert generation
    """
    
    def __init__(self, db_client: AsyncIOMotorClient):
        self.db_client = db_client
        self.db = db_client.ortho_app
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.clinical_validator = ClinicalDataValidator()
        self.ml_quality_assurance = MLDataQualityAssurance()
        
        # Monitoring state
        self.monitoring_sessions: Dict[str, MonitoringConfiguration] = {}
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.anomaly_detectors: Dict[str, Any] = {}
        
        # Performance metrics
        self.processing_times: deque = deque(maxlen=1000)
        self.throughput_counter: int = 0
        self.last_throughput_reset: datetime = datetime.utcnow()
        
        # Alert cooldown tracking
        self.alert_cooldowns: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
        # Status
        self.status: MonitoringStatus = MonitoringStatus.STOPPED
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_monitoring(self, patient_id: str, 
                             config: MonitoringConfiguration = None) -> bool:
        """
        Start real-time monitoring for a patient
        
        Args:
            patient_id: Patient to monitor
            config: Monitoring configuration
            
        Returns:
            Success status
        """
        try:
            if not config:
                config = MonitoringConfiguration(patient_id=patient_id)
            
            # Store configuration
            self.monitoring_sessions[patient_id] = config
            
            # Initialize quality history
            if patient_id not in self.quality_history:
                self.quality_history[patient_id] = deque(maxlen=1000)
            
            # Start monitoring task
            task = asyncio.create_task(self._monitor_patient(patient_id))
            self.monitoring_tasks[patient_id] = task
            
            self.logger.info(f"Started monitoring for patient {patient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring for {patient_id}: {str(e)}")
            return False
    
    async def stop_monitoring(self, patient_id: str) -> bool:
        """Stop monitoring for a patient"""
        try:
            if patient_id in self.monitoring_tasks:
                task = self.monitoring_tasks[patient_id]
                task.cancel()
                del self.monitoring_tasks[patient_id]
            
            if patient_id in self.monitoring_sessions:
                del self.monitoring_sessions[patient_id]
            
            self.logger.info(f"Stopped monitoring for patient {patient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring for {patient_id}: {str(e)}")
            return False
    
    async def process_data_point(self, data: WearableData, 
                               patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single data point in real-time
        
        Args:
            data: Wearable data to process
            patient_context: Patient context for validation
            
        Returns:
            Processing result with quality assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Get monitoring configuration
            config = self.monitoring_sessions.get(data.patient_id)
            if not config:
                return {'error': 'No monitoring session found for patient'}
            
            # Clinical validation
            clinical_report = None
            if config.enable_clinical_monitoring:
                historical_data = await self._get_recent_data(data.patient_id, days=7)
                clinical_report = await self.clinical_validator.validate_wearable_data(
                    data, historical_data, patient_context
                )
            
            # ML quality assessment
            ml_report = None
            if config.enable_ml_monitoring:
                recent_data = await self._get_recent_data(data.patient_id, days=30)
                if recent_data:
                    recent_data.append(data)
                    ml_report = await self.ml_quality_assurance.assess_ml_data_quality(
                        recent_data, 'all', patient_context
                    )
            
            # Generate quality metrics
            quality_metrics = await self._generate_quality_metrics(
                data, clinical_report, ml_report
            )
            
            # Store quality metrics
            self.quality_history[data.patient_id].append(quality_metrics)
            
            # Check for alerts
            alerts = await self._check_quality_alerts(
                data, quality_metrics, clinical_report, ml_report, config
            )
            
            # Store alerts
            for alert in alerts:
                self.alert_history[data.patient_id].append(alert)
                await self._store_alert(alert)
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_times.append(processing_time)
            self.throughput_counter += 1
            
            return {
                'success': True,
                'quality_metrics': quality_metrics,
                'alerts': alerts,
                'processing_time_ms': processing_time * 1000,
                'clinical_report': clinical_report,
                'ml_report': ml_report
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data point: {str(e)}")
            return {'error': str(e)}
    
    async def _monitor_patient(self, patient_id: str):
        """Background monitoring task for a patient"""
        config = self.monitoring_sessions[patient_id]
        
        try:
            while patient_id in self.monitoring_sessions:
                # Check for new data
                await self._check_patient_data(patient_id)
                
                # Wait for next interval
                await asyncio.sleep(config.monitoring_interval)
                
        except asyncio.CancelledError:
            self.logger.info(f"Monitoring cancelled for patient {patient_id}")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop for {patient_id}: {str(e)}")
    
    async def _check_patient_data(self, patient_id: str):
        """Check for new data and process it"""
        try:
            # Get recent data (last 5 minutes)
            recent_data = await self._get_recent_data(patient_id, minutes=5)
            
            if recent_data:
                # Process each data point
                for data in recent_data:
                    await self.process_data_point(data)
                    
        except Exception as e:
            self.logger.error(f"Error checking patient data: {str(e)}")
    
    async def _get_recent_data(self, patient_id: str, 
                             days: int = None, minutes: int = None) -> List[WearableData]:
        """Get recent wearable data for a patient"""
        try:
            # Calculate time threshold
            if days:
                threshold = datetime.utcnow() - timedelta(days=days)
            elif minutes:
                threshold = datetime.utcnow() - timedelta(minutes=minutes)
            else:
                threshold = datetime.utcnow() - timedelta(days=1)
            
            # Query database
            cursor = self.db.wearable_data.find({
                'patient_id': patient_id,
                'date': {'$gte': threshold}
            }).sort('date', -1)
            
            # Convert to WearableData objects
            data_list = []
            async for doc in cursor:
                # Convert MongoDB document to WearableData
                # This would need proper conversion logic
                pass
            
            return data_list
            
        except Exception as e:
            self.logger.error(f"Error getting recent data: {str(e)}")
            return []
    
    async def _generate_quality_metrics(self, data: WearableData, 
                                      clinical_report: DataQualityReport = None,
                                      ml_report: Any = None) -> QualityMetrics:
        """Generate quality metrics for a data point"""
        
        # Calculate overall quality score
        overall_score = 1.0
        completeness_score = 1.0
        consistency_score = 1.0
        reliability_score = 1.0
        clinical_validity_score = 1.0
        ml_readiness_score = 1.0
        
        if clinical_report:
            overall_score = clinical_report.overall_score
            completeness_score = clinical_report.completeness_score
            consistency_score = clinical_report.consistency_score
            reliability_score = clinical_report.reliability_score
            clinical_validity_score = clinical_report.clinical_validity_score
        
        if ml_report:
            ml_readiness_score = ml_report.overall_score
        
        # Calculate anomaly score
        anomaly_score = await self._calculate_anomaly_score(data)
        
        # Calculate trend score
        trend_score = await self._calculate_trend_score(data.patient_id)
        
        # Count recent alerts
        recent_alerts = [a for a in self.alert_history[data.patient_id] 
                        if a.timestamp >= datetime.utcnow() - timedelta(hours=1)]
        
        return QualityMetrics(
            patient_id=data.patient_id,
            timestamp=datetime.utcnow(),
            overall_quality_score=overall_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            reliability_score=reliability_score,
            clinical_validity_score=clinical_validity_score,
            ml_readiness_score=ml_readiness_score,
            anomaly_score=anomaly_score,
            trend_score=trend_score,
            alert_count=len(recent_alerts),
            data_points_processed=len(self.quality_history[data.patient_id])
        )
    
    async def _calculate_anomaly_score(self, data: WearableData) -> float:
        """Calculate anomaly score for data point"""
        try:
            # Get historical data for comparison
            historical_data = await self._get_recent_data(data.patient_id, days=30)
            
            if len(historical_data) < 10:
                return 0.0  # Not enough data for anomaly detection
            
            # Simple anomaly detection based on statistical deviation
            # This would be enhanced with more sophisticated methods
            
            anomaly_score = 0.0
            
            # Check steps anomaly
            if data.activity_metrics and data.activity_metrics.steps:
                historical_steps = [d.activity_metrics.steps for d in historical_data 
                                  if d.activity_metrics and d.activity_metrics.steps]
                if historical_steps:
                    mean_steps = np.mean(historical_steps)
                    std_steps = np.std(historical_steps)
                    if std_steps > 0:
                        z_score = abs(data.activity_metrics.steps - mean_steps) / std_steps
                        anomaly_score = max(anomaly_score, min(z_score / 3.0, 1.0))
            
            # Check heart rate anomaly
            if data.heart_rate_metrics and data.heart_rate_metrics.average_bpm:
                historical_hr = [d.heart_rate_metrics.average_bpm for d in historical_data 
                               if d.heart_rate_metrics and d.heart_rate_metrics.average_bpm]
                if historical_hr:
                    mean_hr = np.mean(historical_hr)
                    std_hr = np.std(historical_hr)
                    if std_hr > 0:
                        z_score = abs(data.heart_rate_metrics.average_bpm - mean_hr) / std_hr
                        anomaly_score = max(anomaly_score, min(z_score / 3.0, 1.0))
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {str(e)}")
            return 0.0
    
    async def _calculate_trend_score(self, patient_id: str) -> float:
        """Calculate trend score based on recent quality history"""
        try:
            recent_metrics = list(self.quality_history[patient_id])[-7:]  # Last 7 data points
            
            if len(recent_metrics) < 3:
                return 0.5  # Neutral trend
            
            # Calculate trend in overall quality
            quality_scores = [m.overall_quality_score for m in recent_metrics]
            
            # Simple linear trend calculation
            x = np.arange(len(quality_scores))
            slope = np.polyfit(x, quality_scores, 1)[0]
            
            # Convert slope to score (0-1, where 0.5 is neutral)
            trend_score = 0.5 + np.tanh(slope * 10) * 0.5
            
            return max(0.0, min(1.0, trend_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend score: {str(e)}")
            return 0.5
    
    async def _check_quality_alerts(self, data: WearableData, 
                                  quality_metrics: QualityMetrics,
                                  clinical_report: DataQualityReport = None,
                                  ml_report: Any = None,
                                  config: MonitoringConfiguration = None) -> List[QualityAlert]:
        """Check for quality alerts based on current data"""
        alerts = []
        
        try:
            # Check overall quality threshold
            if quality_metrics.overall_quality_score < config.quality_thresholds['critical']:
                if not self._is_alert_in_cooldown(data.patient_id, 'critical_quality'):
                    alerts.append(QualityAlert(
                        alert_id=f"critical_quality_{data.patient_id}_{datetime.utcnow().isoformat()}",
                        patient_id=data.patient_id,
                        timestamp=datetime.utcnow(),
                        priority=AlertPriority.CRITICAL,
                        category="data_quality",
                        message=f"Critical data quality detected: {quality_metrics.overall_quality_score:.2f}",
                        data_point={"quality_score": quality_metrics.overall_quality_score},
                        quality_score=quality_metrics.overall_quality_score,
                        recommended_action="Immediate review required - data may be unreliable",
                        clinical_impact="High - may affect clinical decisions"
                    ))
            
            # Check anomaly score
            if quality_metrics.anomaly_score > config.anomaly_sensitivity:
                if not self._is_alert_in_cooldown(data.patient_id, 'anomaly'):
                    alerts.append(QualityAlert(
                        alert_id=f"anomaly_{data.patient_id}_{datetime.utcnow().isoformat()}",
                        patient_id=data.patient_id,
                        timestamp=datetime.utcnow(),
                        priority=AlertPriority.HIGH,
                        category="anomaly_detection",
                        message=f"Data anomaly detected: score {quality_metrics.anomaly_score:.2f}",
                        data_point={"anomaly_score": quality_metrics.anomaly_score},
                        quality_score=quality_metrics.overall_quality_score,
                        recommended_action="Review data pattern and patient status",
                        clinical_impact="Medium - unusual pattern detected"
                    ))
            
            # Check clinical validation issues
            if clinical_report and clinical_report.requires_review:
                critical_issues = [r for r in clinical_report.validation_results 
                                 if r.severity == 'critical']
                if critical_issues and not self._is_alert_in_cooldown(data.patient_id, 'clinical_critical'):
                    alerts.append(QualityAlert(
                        alert_id=f"clinical_critical_{data.patient_id}_{datetime.utcnow().isoformat()}",
                        patient_id=data.patient_id,
                        timestamp=datetime.utcnow(),
                        priority=AlertPriority.CRITICAL,
                        category="clinical_validation",
                        message=f"Critical clinical validation issues: {len(critical_issues)} found",
                        data_point={"validation_issues": len(critical_issues)},
                        quality_score=quality_metrics.overall_quality_score,
                        recommended_action="Clinical review required immediately",
                        clinical_impact="High - may indicate patient safety concern"
                    ))
            
            # Check ML model readiness
            if ml_report and not ml_report.ml_model_ready:
                if not self._is_alert_in_cooldown(data.patient_id, 'ml_readiness'):
                    alerts.append(QualityAlert(
                        alert_id=f"ml_readiness_{data.patient_id}_{datetime.utcnow().isoformat()}",
                        patient_id=data.patient_id,
                        timestamp=datetime.utcnow(),
                        priority=AlertPriority.MEDIUM,
                        category="ml_quality",
                        message="Data not suitable for ML model predictions",
                        data_point={"ml_ready": False},
                        quality_score=quality_metrics.overall_quality_score,
                        recommended_action="Improve data collection consistency",
                        clinical_impact="Low - affects predictive analytics only"
                    ))
            
            # Set cooldowns for triggered alerts
            for alert in alerts:
                self._set_alert_cooldown(alert.patient_id, alert.category, config.alert_cooldown)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking quality alerts: {str(e)}")
            return []
    
    def _is_alert_in_cooldown(self, patient_id: str, alert_type: str) -> bool:
        """Check if alert type is in cooldown period"""
        if patient_id not in self.alert_cooldowns:
            return False
        
        if alert_type not in self.alert_cooldowns[patient_id]:
            return False
        
        cooldown_end = self.alert_cooldowns[patient_id][alert_type]
        return datetime.utcnow() < cooldown_end
    
    def _set_alert_cooldown(self, patient_id: str, alert_type: str, cooldown_seconds: int):
        """Set cooldown period for alert type"""
        if patient_id not in self.alert_cooldowns:
            self.alert_cooldowns[patient_id] = {}
        
        self.alert_cooldowns[patient_id][alert_type] = datetime.utcnow() + timedelta(seconds=cooldown_seconds)
    
    async def _store_alert(self, alert: QualityAlert):
        """Store alert in database"""
        try:
            alert_doc = {
                'alert_id': alert.alert_id,
                'patient_id': alert.patient_id,
                'timestamp': alert.timestamp,
                'priority': alert.priority,
                'category': alert.category,
                'message': alert.message,
                'data_point': alert.data_point,
                'quality_score': alert.quality_score,
                'recommended_action': alert.recommended_action,
                'clinical_impact': alert.clinical_impact,
                'auto_resolved': alert.auto_resolved,
                'resolution_timestamp': alert.resolution_timestamp
            }
            
            await self.db.quality_alerts.insert_one(alert_doc)
            
        except Exception as e:
            self.logger.error(f"Error storing alert: {str(e)}")
    
    async def get_quality_dashboard(self, patient_id: str = None) -> Dict[str, Any]:
        """Get quality monitoring dashboard data"""
        try:
            if patient_id:
                # Patient-specific dashboard
                return await self._get_patient_dashboard(patient_id)
            else:
                # System-wide dashboard
                return await self._get_system_dashboard()
                
        except Exception as e:
            self.logger.error(f"Error getting quality dashboard: {str(e)}")
            return {'error': str(e)}
    
    async def _get_patient_dashboard(self, patient_id: str) -> Dict[str, Any]:
        """Get patient-specific quality dashboard"""
        
        # Get recent quality metrics
        recent_metrics = list(self.quality_history[patient_id])[-24:]  # Last 24 data points
        
        # Get recent alerts
        recent_alerts = list(self.alert_history[patient_id])[-10:]  # Last 10 alerts
        
        # Calculate averages
        if recent_metrics:
            avg_quality = np.mean([m.overall_quality_score for m in recent_metrics])
            avg_completeness = np.mean([m.completeness_score for m in recent_metrics])
            avg_consistency = np.mean([m.consistency_score for m in recent_metrics])
        else:
            avg_quality = avg_completeness = avg_consistency = 0.0
        
        return {
            'patient_id': patient_id,
            'monitoring_status': 'active' if patient_id in self.monitoring_sessions else 'inactive',
            'quality_metrics': {
                'average_quality_score': avg_quality,
                'average_completeness': avg_completeness,
                'average_consistency': avg_consistency,
                'data_points_processed': len(self.quality_history[patient_id])
            },
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp,
                    'priority': alert.priority,
                    'category': alert.category,
                    'message': alert.message
                }
                for alert in recent_alerts
            ],
            'quality_trend': [
                {
                    'timestamp': metric.timestamp,
                    'quality_score': metric.overall_quality_score,
                    'anomaly_score': metric.anomaly_score
                }
                for metric in recent_metrics
            ]
        }
    
    async def _get_system_dashboard(self) -> Dict[str, Any]:
        """Get system-wide quality dashboard"""
        
        # Calculate system metrics
        total_patients = len(self.monitoring_sessions)
        total_alerts = sum(len(alerts) for alerts in self.alert_history.values())
        
        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        current_throughput = self.throughput_counter / max(1, (datetime.utcnow() - self.last_throughput_reset).total_seconds())
        
        # Quality distribution
        all_recent_metrics = []
        for patient_metrics in self.quality_history.values():
            all_recent_metrics.extend(list(patient_metrics)[-10:])  # Last 10 per patient
        
        if all_recent_metrics:
            quality_distribution = {
                'excellent': len([m for m in all_recent_metrics if m.overall_quality_score >= 0.9]),
                'good': len([m for m in all_recent_metrics if 0.8 <= m.overall_quality_score < 0.9]),
                'fair': len([m for m in all_recent_metrics if 0.7 <= m.overall_quality_score < 0.8]),
                'poor': len([m for m in all_recent_metrics if m.overall_quality_score < 0.7])
            }
        else:
            quality_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        return {
            'system_status': self.status,
            'monitoring_sessions': total_patients,
            'total_alerts': total_alerts,
            'performance': {
                'avg_processing_time_ms': avg_processing_time * 1000,
                'throughput_per_second': current_throughput,
                'data_points_processed': sum(len(metrics) for metrics in self.quality_history.values())
            },
            'quality_distribution': quality_distribution,
            'active_patients': list(self.monitoring_sessions.keys())
        }