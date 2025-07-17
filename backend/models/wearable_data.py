"""
Comprehensive wearable data models for orthopedic recovery tracking.
Supports HealthKit, Android Health, and other wearable device integrations.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date, time
from enum import Enum
import uuid


class DataSource(str, Enum):
    """Data source types for wearable data"""
    HEALTHKIT = "HealthKit"
    GOOGLE_FIT = "Google Fit"
    FITBIT = "Fitbit"
    SAMSUNG_HEALTH = "Samsung Health"
    GARMIN = "Garmin"
    MANUAL = "Manual Entry"
    UNKNOWN = "Unknown"


class DataQuality(str, Enum):
    """Data quality indicators"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    POOR = "Poor"


class ActivityType(str, Enum):
    """Types of physical activities"""
    WALKING = "Walking"
    RUNNING = "Running"
    CYCLING = "Cycling"
    SWIMMING = "Swimming"
    STRENGTH_TRAINING = "Strength Training"
    PHYSICAL_THERAPY = "Physical Therapy"
    YOGA = "Yoga"
    STAIRS = "Stairs"
    OTHER = "Other"


class SleepStage(str, Enum):
    """Sleep stages for detailed sleep analysis"""
    AWAKE = "Awake"
    LIGHT = "Light"
    DEEP = "Deep"
    REM = "REM"


# Activity Data Models
class ActivityMetrics(BaseModel):
    """Daily activity metrics"""
    steps: Optional[int] = Field(None, ge=0, le=100000)
    distance_meters: Optional[float] = Field(None, ge=0, le=100000)
    floors_climbed: Optional[int] = Field(None, ge=0, le=1000)
    calories_active: Optional[float] = Field(None, ge=0, le=10000)
    calories_total: Optional[float] = Field(None, ge=0, le=10000)
    active_minutes: Optional[int] = Field(None, ge=0, le=1440)
    sedentary_minutes: Optional[int] = Field(None, ge=0, le=1440)
    
    @validator('sedentary_minutes', 'active_minutes')
    def validate_daily_minutes(cls, v, values):
        if v is not None and 'active_minutes' in values:
            total = v + (values.get('active_minutes') or 0)
            if total > 1440:  # 24 hours
                raise ValueError('Total active + sedentary minutes cannot exceed 1440')
        return v


class ExerciseSession(BaseModel):
    """Individual exercise session data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: datetime
    activity_type: ActivityType
    duration_minutes: int = Field(ge=1, le=1440)
    calories_burned: Optional[float] = Field(None, ge=0)
    average_heart_rate: Optional[int] = Field(None, ge=30, le=220)
    max_heart_rate: Optional[int] = Field(None, ge=30, le=220)
    distance_meters: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


# Heart Rate Data Models
class HeartRateMetrics(BaseModel):
    """Comprehensive heart rate data"""
    resting_hr: Optional[int] = Field(None, ge=30, le=120)
    average_hr: Optional[int] = Field(None, ge=30, le=220)
    max_hr: Optional[int] = Field(None, ge=30, le=220)
    min_hr: Optional[int] = Field(None, ge=30, le=220)
    hr_variability_ms: Optional[float] = Field(None, ge=0, le=200)
    recovery_hr: Optional[int] = Field(None, ge=30, le=220)
    
    @validator('max_hr')
    def validate_max_hr(cls, v, values):
        if v is not None and 'min_hr' in values and values['min_hr'] is not None:
            if v <= values['min_hr']:
                raise ValueError('Max heart rate must be greater than min heart rate')
        return v


class HeartRateZone(BaseModel):
    """Time spent in different heart rate zones"""
    zone_1_minutes: Optional[int] = Field(None, ge=0)  # 50-60% max HR
    zone_2_minutes: Optional[int] = Field(None, ge=0)  # 60-70% max HR
    zone_3_minutes: Optional[int] = Field(None, ge=0)  # 70-80% max HR
    zone_4_minutes: Optional[int] = Field(None, ge=0)  # 80-90% max HR
    zone_5_minutes: Optional[int] = Field(None, ge=0)  # 90-100% max HR


# Sleep Data Models
class SleepMetrics(BaseModel):
    """Comprehensive sleep analysis"""
    total_sleep_minutes: Optional[int] = Field(None, ge=0, le=1440)
    sleep_efficiency: Optional[float] = Field(None, ge=0, le=100)
    time_to_fall_asleep_minutes: Optional[int] = Field(None, ge=0, le=480)
    wake_up_count: Optional[int] = Field(None, ge=0, le=50)
    time_awake_during_sleep_minutes: Optional[int] = Field(None, ge=0, le=480)
    
    # Sleep stages
    light_sleep_minutes: Optional[int] = Field(None, ge=0, le=1440)
    deep_sleep_minutes: Optional[int] = Field(None, ge=0, le=1440)
    rem_sleep_minutes: Optional[int] = Field(None, ge=0, le=1440)
    
    # Sleep timing
    bedtime: Optional[time] = None
    wake_time: Optional[time] = None
    
    @validator('sleep_efficiency')
    def validate_sleep_efficiency(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Sleep efficiency must be between 0 and 100 percent')
        return v


class SleepSession(BaseModel):
    """Individual sleep session with detailed stages"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: datetime
    sleep_stages: List[Dict[str, Union[datetime, SleepStage]]] = []
    sleep_score: Optional[int] = Field(None, ge=0, le=100)


# Movement and Mobility Models
class MovementMetrics(BaseModel):
    """Movement patterns and mobility indicators"""
    walking_speed_ms: Optional[float] = Field(None, ge=0, le=10)  # meters per second
    walking_asymmetry: Optional[float] = Field(None, ge=0, le=100)  # percentage
    step_length_cm: Optional[float] = Field(None, ge=0, le=200)
    cadence_steps_per_minute: Optional[int] = Field(None, ge=0, le=300)
    balance_score: Optional[float] = Field(None, ge=0, le=100)
    mobility_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Gait analysis
    stance_time_ms: Optional[float] = Field(None, ge=0, le=2000)
    swing_time_ms: Optional[float] = Field(None, ge=0, le=2000)
    double_support_time_ms: Optional[float] = Field(None, ge=0, le=1000)


# Environmental and Context Data
class EnvironmentalData(BaseModel):
    """Environmental factors affecting recovery"""
    temperature_celsius: Optional[float] = Field(None, ge=-50, le=60)
    humidity_percent: Optional[float] = Field(None, ge=0, le=100)
    air_quality_index: Optional[int] = Field(None, ge=0, le=500)
    altitude_meters: Optional[float] = Field(None, ge=-500, le=9000)


# Physiological Metrics
class PhysiologicalMetrics(BaseModel):
    """Additional physiological measurements"""
    oxygen_saturation: Optional[float] = Field(None, ge=70, le=100)
    blood_pressure_systolic: Optional[int] = Field(None, ge=60, le=250)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150)
    respiratory_rate: Optional[int] = Field(None, ge=8, le=40)
    body_temperature_celsius: Optional[float] = Field(None, ge=35, le=42)
    
    @validator('blood_pressure_diastolic')
    def validate_blood_pressure(cls, v, values):
        if v is not None and 'blood_pressure_systolic' in values:
            systolic = values.get('blood_pressure_systolic')
            if systolic is not None and v >= systolic:
                raise ValueError('Diastolic pressure must be less than systolic pressure')
        return v


# Data Quality and Metadata
class DataMetadata(BaseModel):
    """Metadata for data quality and source tracking"""
    source: DataSource = DataSource.UNKNOWN
    device_model: Optional[str] = None
    app_version: Optional[str] = None
    data_quality: DataQuality = DataQuality.MEDIUM
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    processing_notes: Optional[str] = None
    sync_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Main Comprehensive Wearable Data Model
class ComprehensiveWearableData(BaseModel):
    """Complete wearable data record for a specific date"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: date
    
    # Core metrics
    activity_metrics: Optional[ActivityMetrics] = None
    heart_rate_metrics: Optional[HeartRateMetrics] = None
    heart_rate_zones: Optional[HeartRateZone] = None
    sleep_metrics: Optional[SleepMetrics] = None
    movement_metrics: Optional[MovementMetrics] = None
    physiological_metrics: Optional[PhysiologicalMetrics] = None
    environmental_data: Optional[EnvironmentalData] = None
    
    # Sessions and detailed data
    exercise_sessions: List[ExerciseSession] = []
    sleep_sessions: List[SleepSession] = []
    
    # Metadata
    data_metadata: DataMetadata = Field(default_factory=DataMetadata)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Create/Update models for API
class ComprehensiveWearableDataCreate(BaseModel):
    """Create model for comprehensive wearable data"""
    patient_id: str
    date: date
    activity_metrics: Optional[ActivityMetrics] = None
    heart_rate_metrics: Optional[HeartRateMetrics] = None
    heart_rate_zones: Optional[HeartRateZone] = None
    sleep_metrics: Optional[SleepMetrics] = None
    movement_metrics: Optional[MovementMetrics] = None
    physiological_metrics: Optional[PhysiologicalMetrics] = None
    environmental_data: Optional[EnvironmentalData] = None
    exercise_sessions: List[ExerciseSession] = []
    sleep_sessions: List[SleepSession] = []
    data_metadata: Optional[DataMetadata] = None


class BulkWearableDataImport(BaseModel):
    """Model for bulk importing wearable data"""
    patient_id: str
    data_records: List[ComprehensiveWearableDataCreate]
    import_source: DataSource = DataSource.UNKNOWN
    import_notes: Optional[str] = None


# Analytics and Trend Models
class WearableDataTrends(BaseModel):
    """Trends analysis for wearable data"""
    metric_name: str
    patient_id: str
    start_date: date
    end_date: date
    values: List[float]
    dates: List[date]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_significance: float  # 0-1 significance score
    correlation_with_pro_scores: Optional[float] = None


class ActivityPattern(BaseModel):
    """Activity pattern analysis"""
    patient_id: str
    pattern_type: str  # "daily", "weekly", "recovery_phase"
    average_steps: Optional[float] = None
    average_active_minutes: Optional[float] = None
    peak_activity_hour: Optional[int] = None
    consistency_score: Optional[float] = None  # 0-1 score
    identified_at: datetime = Field(default_factory=datetime.utcnow)


# Recovery-specific Analytics
class RecoveryIndicators(BaseModel):
    """Wearable-based recovery indicators"""
    patient_id: str
    analysis_date: date
    
    # Activity trends
    walking_speed_trend: Optional[str] = None  # "improving", "stable", "declining"
    activity_consistency: Optional[float] = None  # 0-1 score
    stairs_climbing_progress: Optional[float] = None  # vs baseline
    
    # Sleep quality indicators
    sleep_quality_trend: Optional[str] = None
    sleep_efficiency_avg: Optional[float] = None
    
    # Heart rate recovery
    resting_hr_trend: Optional[str] = None
    hrv_trend: Optional[str] = None
    
    # Risk indicators
    sedentary_time_alert: bool = False
    activity_drop_alert: bool = False
    sleep_disruption_alert: bool = False
    
    # Correlations with clinical outcomes
    activity_pain_correlation: Optional[float] = None
    sleep_pro_score_correlation: Optional[float] = None