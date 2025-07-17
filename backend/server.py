from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uuid
from datetime import datetime, date, timedelta
from enum import Enum
from recovery_trajectories import (
    get_trajectory_for_diagnosis,
    get_expected_score_at_week,
    calculate_weeks_post_surgery,
    get_recovery_status_from_trajectory,
    get_milestone_status,
    TrajectoryPoint
)
from predictive_models import RecoveryPredictor
from diagnosis_recommendations import RecommendationEngine
from models.wearable_data import (
    ComprehensiveWearableData,
    ComprehensiveWearableDataCreate,
    BulkWearableDataImport,
    WearableDataTrends,
    RecoveryIndicators,
    ActivityMetrics,
    HeartRateMetrics,
    SleepMetrics,
    MovementMetrics,
    DataSource,
    DataQuality
)
from schemas.wearable_schemas import (
    WearableDataSchemas,
    WearableDataAggregations,
    WearableDataQueries,
    initialize_wearable_schemas
)
from routers.wearable_api import wearable_router, set_database

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Enums for data validation
class DiagnosisType(str, Enum):
    # Knee Diagnoses (use KOOS)
    ACL_TEAR = "ACL Tear"
    MENISCUS_TEAR = "Meniscus Tear"
    CARTILAGE_DEFECT = "Cartilage Defect"
    KNEE_OSTEOARTHRITIS = "Knee Osteoarthritis"
    POST_TOTAL_KNEE_REPLACEMENT = "Post Total Knee Replacement"
    
    # Shoulder Diagnoses (use ASES)
    ROTATOR_CUFF_TEAR = "Rotator Cuff Tear"
    LABRAL_TEAR = "Labral Tear"
    SHOULDER_INSTABILITY = "Shoulder Instability"
    SHOULDER_OSTEOARTHRITIS = "Shoulder Osteoarthritis"
    POST_TOTAL_SHOULDER_REPLACEMENT = "Post Total Shoulder Replacement"

class BodyPart(str, Enum):
    KNEE = "KNEE"
    SHOULDER = "SHOULDER"

class RecoveryStatus(str, Enum):
    ON_TRACK = "On Track"
    AT_RISK = "At Risk"
    NEEDS_ATTENTION = "Needs Attention"

# Helper function to determine body part from diagnosis
def get_body_part(diagnosis: DiagnosisType) -> BodyPart:
    knee_diagnoses = {
        DiagnosisType.ACL_TEAR,
        DiagnosisType.MENISCUS_TEAR,
        DiagnosisType.CARTILAGE_DEFECT,
        DiagnosisType.KNEE_OSTEOARTHRITIS,
        DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT
    }
    
    if diagnosis in knee_diagnoses:
        return BodyPart.KNEE
    else:
        return BodyPart.SHOULDER

# Define Models
class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    diagnosis_type: DiagnosisType
    date_of_injury: datetime
    date_of_surgery: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def body_part(self) -> BodyPart:
        return get_body_part(self.diagnosis_type)

class PatientCreate(BaseModel):
    name: str
    email: str
    diagnosis_type: DiagnosisType
    date_of_injury: datetime
    date_of_surgery: Optional[datetime] = None

# Legacy WearableData models for backward compatibility
class WearableData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    steps: int
    heart_rate: int
    oxygen_saturation: float
    sleep_hours: float
    walking_speed: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class WearableDataCreate(BaseModel):
    patient_id: str
    date: datetime
    steps: int
    heart_rate: int
    oxygen_saturation: float
    sleep_hours: float
    walking_speed: Optional[float] = None

class Survey(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    pain_score: int = Field(ge=0, le=10)  # 0-10 scale
    mobility_score: int = Field(ge=0, le=10)  # 0-10 scale
    activities_of_daily_living: Dict[str, int] = {}  # Various activities and their scores
    range_of_motion: Dict[str, float] = {}  # Different ROM measurements
    strength: Dict[str, int] = {}  # Different strength measurements
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SurveyCreate(BaseModel):
    patient_id: str
    date: datetime
    pain_score: int = Field(ge=0, le=10)
    mobility_score: int = Field(ge=0, le=10)
    activities_of_daily_living: Dict[str, int] = {}
    range_of_motion: Dict[str, float] = {}
    strength: Dict[str, int] = {}
    notes: Optional[str] = None

class AIInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    recovery_status: RecoveryStatus
    recommendations: List[str] = []
    risk_factors: List[str] = []
    progress_percentage: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class RecommendationItem(BaseModel):
    text: str
    priority: str = Field(default="medium")  # "high", "medium", "low"
    evidence: Optional[str] = None
    category: Optional[str] = None  # "pain", "function", "activity", "general"

class TrajectoryAnalysis(BaseModel):
    subscale: str
    actual_score: float
    expected_score: float
    lower_bound: float
    upper_bound: float
    status: str  # "Ahead of Schedule", "On Track", "Slightly Behind", "Behind Schedule"
    deviation_percentage: float

class RecoveryMilestoneStatus(BaseModel):
    week: int
    description: str
    expected_score: float
    actual_score: float
    achieved: bool
    critical: bool
    subscale: str

class RiskAssessment(BaseModel):
    risk_score: float = Field(ge=0, le=100)  # 0-100 risk score
    risk_category: str  # "Low", "Moderate", "High", "Very High"
    risk_factors: List[str] = []
    protective_factors: List[str] = []

class RecoveryVelocity(BaseModel):
    subscale: str
    current_score: float
    previous_score: Optional[float] = None
    velocity: Optional[float] = None  # Points per week
    trend: str  # "Improving", "Stable", "Declining"

class EnhancedAIInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    weeks_post_surgery: int
    recovery_status: RecoveryStatus
    recommendations: List[RecommendationItem] = []
    risk_factors: List[str] = []
    progress_percentage: float = 0.0
    
    # New enhanced fields
    trajectory_analysis: List[TrajectoryAnalysis] = []
    milestone_status: List[RecoveryMilestoneStatus] = []
    risk_assessment: Optional[RiskAssessment] = None
    recovery_velocity: List[RecoveryVelocity] = []
    projected_recovery_date: Optional[datetime] = None
    confidence_interval: Optional[str] = None
    
    # Key findings and patterns
    key_findings: List[str] = []
    concerning_patterns: List[str] = []
    positive_trends: List[str] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# KOOS (Knee injury and Osteoarthritis Outcome Score) Models
class KOOSResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    # Symptoms subscale (7 items)
    s1_swelling: int = Field(ge=0, le=4)  # Do you have swelling in your knee?
    s2_grinding: int = Field(ge=0, le=4)  # Do you feel grinding, hear clicking or any other type of noise when your knee moves?
    s3_catching: int = Field(ge=0, le=4)  # Does your knee catch or hang up when moving?
    s4_straighten: int = Field(ge=0, le=4)  # Can you straighten your knee fully?
    s5_bend: int = Field(ge=0, le=4)  # Can you bend your knee fully?
    s6_stiffness_morning: int = Field(ge=0, le=4)  # How severe is your knee stiffness after first wakening in the morning?
    s7_stiffness_later: int = Field(ge=0, le=4)  # How severe is your knee stiffness after sitting, lying or resting later in the day?
    # Pain subscale (9 items)
    p1_frequency: int = Field(ge=0, le=4)  # How often do you experience knee pain?
    p2_twisting: int = Field(ge=0, le=4)  # Twisting/pivoting on your knee
    p3_straightening: int = Field(ge=0, le=4)  # Straightening knee fully
    p4_bending: int = Field(ge=0, le=4)  # Bending knee fully
    p5_walking_flat: int = Field(ge=0, le=4)  # Walking on flat surface
    p6_stairs: int = Field(ge=0, le=4)  # Going up or down stairs
    p7_night: int = Field(ge=0, le=4)  # At night while in bed
    p8_sitting: int = Field(ge=0, le=4)  # Sitting or lying
    p9_standing: int = Field(ge=0, le=4)  # Standing upright
    # ADL subscale (17 items)
    a1_descending_stairs: int = Field(ge=0, le=4)  # Descending stairs
    a2_ascending_stairs: int = Field(ge=0, le=4)  # Ascending stairs
    a3_rising_sitting: int = Field(ge=0, le=4)  # Rising from sitting
    a4_standing: int = Field(ge=0, le=4)  # Standing
    a5_bending_floor: int = Field(ge=0, le=4)  # Bending to floor/pick up an object
    a6_walking_flat: int = Field(ge=0, le=4)  # Walking on flat surface
    a7_car: int = Field(ge=0, le=4)  # Getting in/out of car
    a8_shopping: int = Field(ge=0, le=4)  # Going shopping
    a9_socks_on: int = Field(ge=0, le=4)  # Putting on socks/stockings
    a10_rising_bed: int = Field(ge=0, le=4)  # Rising from bed
    a11_socks_off: int = Field(ge=0, le=4)  # Taking off socks/stockings
    a12_lying_bed: int = Field(ge=0, le=4)  # Lying in bed (turning over, maintaining knee position)
    a13_bath: int = Field(ge=0, le=4)  # Getting in/out of bath
    a14_sitting: int = Field(ge=0, le=4)  # Sitting
    a15_toilet: int = Field(ge=0, le=4)  # Getting on/off toilet
    a16_heavy_duties: int = Field(ge=0, le=4)  # Heavy domestic duties (moving heavy boxes, scrubbing floors, etc)
    a17_light_duties: int = Field(ge=0, le=4)  # Light domestic duties (cooking, dusting, etc)
    # Sport/Recreation subscale (5 items)
    sp1_squatting: int = Field(ge=0, le=4)  # Squatting
    sp2_running: int = Field(ge=0, le=4)  # Running
    sp3_jumping: int = Field(ge=0, le=4)  # Jumping
    sp4_twisting: int = Field(ge=0, le=4)  # Twisting/pivoting on your injured knee
    sp5_kneeling: int = Field(ge=0, le=4)  # Kneeling
    # Quality of Life subscale (4 items)
    q1_awareness: int = Field(ge=0, le=4)  # How often are you aware of your knee problem?
    q2_lifestyle: int = Field(ge=0, le=4)  # Have you modified your life style to avoid potentially damaging activities to your knee?
    q3_confidence: int = Field(ge=0, le=4)  # How much are you troubled with lack of confidence in your knee?
    q4_difficulty: int = Field(ge=0, le=4)  # In general, how much difficulty do you have with your knee?
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class KOOSScores(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    koos_response_id: str
    symptoms_score: float  # 0-100 (100 = no problems)
    pain_score: float  # 0-100 (100 = no problems)
    adl_score: float  # 0-100 (100 = no problems)
    sport_score: float  # 0-100 (100 = no problems)
    qol_score: float  # 0-100 (100 = no problems)
    total_score: float  # Average of all subscales
    is_complete: bool  # False if too many missing responses
    created_at: datetime = Field(default_factory=datetime.utcnow)

class KOOSCreate(BaseModel):
    patient_id: str
    date: datetime
    s1_swelling: int = Field(ge=0, le=4)
    s2_grinding: int = Field(ge=0, le=4)
    s3_catching: int = Field(ge=0, le=4)
    s4_straighten: int = Field(ge=0, le=4)
    s5_bend: int = Field(ge=0, le=4)
    s6_stiffness_morning: int = Field(ge=0, le=4)
    s7_stiffness_later: int = Field(ge=0, le=4)
    p1_frequency: int = Field(ge=0, le=4)
    p2_twisting: int = Field(ge=0, le=4)
    p3_straightening: int = Field(ge=0, le=4)
    p4_bending: int = Field(ge=0, le=4)
    p5_walking_flat: int = Field(ge=0, le=4)
    p6_stairs: int = Field(ge=0, le=4)
    p7_night: int = Field(ge=0, le=4)
    p8_sitting: int = Field(ge=0, le=4)
    p9_standing: int = Field(ge=0, le=4)
    a1_descending_stairs: int = Field(ge=0, le=4)
    a2_ascending_stairs: int = Field(ge=0, le=4)
    a3_rising_sitting: int = Field(ge=0, le=4)
    a4_standing: int = Field(ge=0, le=4)
    a5_bending_floor: int = Field(ge=0, le=4)
    a6_walking_flat: int = Field(ge=0, le=4)
    a7_car: int = Field(ge=0, le=4)
    a8_shopping: int = Field(ge=0, le=4)
    a9_socks_on: int = Field(ge=0, le=4)
    a10_rising_bed: int = Field(ge=0, le=4)
    a11_socks_off: int = Field(ge=0, le=4)
    a12_lying_bed: int = Field(ge=0, le=4)
    a13_bath: int = Field(ge=0, le=4)
    a14_sitting: int = Field(ge=0, le=4)
    a15_toilet: int = Field(ge=0, le=4)
    a16_heavy_duties: int = Field(ge=0, le=4)
    a17_light_duties: int = Field(ge=0, le=4)
    sp1_squatting: int = Field(ge=0, le=4)
    sp2_running: int = Field(ge=0, le=4)
    sp3_jumping: int = Field(ge=0, le=4)
    sp4_twisting: int = Field(ge=0, le=4)
    sp5_kneeling: int = Field(ge=0, le=4)
    q1_awareness: int = Field(ge=0, le=4)
    q2_lifestyle: int = Field(ge=0, le=4)
    q3_confidence: int = Field(ge=0, le=4)
    q4_difficulty: int = Field(ge=0, le=4)

# ASES (American Shoulder and Elbow Surgeons) Models
class ASESResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    # Pain component (0-10 VAS)
    pain_vas: float = Field(ge=0, le=10)  # How bad is your pain today?
    # Function component (10 items, 0-3 each)
    f1_coat: int = Field(ge=0, le=3)  # Put on a coat
    f2_sleep: int = Field(ge=0, le=3)  # Sleep on your painful or affected side
    f3_wash_back: int = Field(ge=0, le=3)  # Wash your back or do up bra in back
    f4_toileting: int = Field(ge=0, le=3)  # Manage toileting
    f5_comb_hair: int = Field(ge=0, le=3)  # Comb your hair
    f6_high_shelf: int = Field(ge=0, le=3)  # Reach a high shelf
    f7_lift_10lbs: int = Field(ge=0, le=3)  # Lift 10 pounds above shoulder level
    f8_throw_ball: int = Field(ge=0, le=3)  # Throw a ball overhand
    f9_usual_work: int = Field(ge=0, le=3)  # Do usual work
    f10_usual_sport: int = Field(ge=0, le=3)  # Do usual sport
    # Additional tracking (not scored)
    has_instability: bool  # Do you have instability (feeling of shoulder coming out of joint)?
    instability_severity: Optional[float] = Field(None, ge=0, le=10)  # If yes, rate instability 0-10
    usual_work_description: Optional[str] = None  # Description of usual work
    usual_sport_description: Optional[str] = None  # Description of usual sport
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ASESScores(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: datetime
    ases_response_id: str
    pain_component: float  # 0-50 points
    function_component: float  # 0-50 points
    total_score: float  # 0-100 points (100 = best)
    is_complete: bool  # False if any required responses missing
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ASESCreate(BaseModel):
    patient_id: str
    date: datetime
    pain_vas: float = Field(ge=0, le=10)
    f1_coat: int = Field(ge=0, le=3)
    f2_sleep: int = Field(ge=0, le=3)
    f3_wash_back: int = Field(ge=0, le=3)
    f4_toileting: int = Field(ge=0, le=3)
    f5_comb_hair: int = Field(ge=0, le=3)
    f6_high_shelf: int = Field(ge=0, le=3)
    f7_lift_10lbs: int = Field(ge=0, le=3)
    f8_throw_ball: int = Field(ge=0, le=3)
    f9_usual_work: int = Field(ge=0, le=3)
    f10_usual_sport: int = Field(ge=0, le=3)
    has_instability: bool
    instability_severity: Optional[float] = Field(None, ge=0, le=10)
    usual_work_description: Optional[str] = None
    usual_sport_description: Optional[str] = None

# Scoring Algorithm Functions
def calculate_koos_scores(response_data: dict) -> dict:
    """
    Calculate KOOS subscale scores according to official scoring methodology.
    Each subscale: 100 - (mean of items × 25)
    100 = no problems, 0 = extreme problems
    """
    # Define subscale items
    symptoms_items = ['s1_swelling', 's2_grinding', 's3_catching', 's4_straighten', 's5_bend', 's6_stiffness_morning', 's7_stiffness_later']
    pain_items = ['p1_frequency', 'p2_twisting', 'p3_straightening', 'p4_bending', 'p5_walking_flat', 'p6_stairs', 'p7_night', 'p8_sitting', 'p9_standing']
    adl_items = ['a1_descending_stairs', 'a2_ascending_stairs', 'a3_rising_sitting', 'a4_standing', 'a5_bending_floor', 'a6_walking_flat', 'a7_car', 'a8_shopping', 'a9_socks_on', 'a10_rising_bed', 'a11_socks_off', 'a12_lying_bed', 'a13_bath', 'a14_sitting', 'a15_toilet', 'a16_heavy_duties', 'a17_light_duties']
    sport_items = ['sp1_squatting', 'sp2_running', 'sp3_jumping', 'sp4_twisting', 'sp5_kneeling']
    qol_items = ['q1_awareness', 'q2_lifestyle', 'q3_confidence', 'q4_difficulty']
    
    def calculate_subscale_score(items: List[str], max_missing: int = 2) -> Optional[float]:
        """Calculate subscale score with missing data handling"""
        values = []
        for item in items:
            if item in response_data and response_data[item] is not None:
                values.append(response_data[item])
        
        # Check if too many missing values
        missing_count = len(items) - len(values)
        if missing_count > max_missing:
            return None
        
        # Calculate mean and convert to 0-100 scale
        if values:
            mean_score = sum(values) / len(values)
            return 100 - (mean_score * 25)
        return None
    
    # Calculate each subscale
    symptoms_score = calculate_subscale_score(symptoms_items)
    pain_score = calculate_subscale_score(pain_items)
    adl_score = calculate_subscale_score(adl_items)
    sport_score = calculate_subscale_score(sport_items)
    qol_score = calculate_subscale_score(qol_items)
    
    # Calculate total score (average of completed subscales)
    completed_scores = [score for score in [symptoms_score, pain_score, adl_score, sport_score, qol_score] if score is not None]
    total_score = sum(completed_scores) / len(completed_scores) if completed_scores else None
    
    # Check if questionnaire is complete enough
    is_complete = all(score is not None for score in [symptoms_score, pain_score, adl_score, sport_score, qol_score])
    
    return {
        'symptoms_score': symptoms_score or 0.0,
        'pain_score': pain_score or 0.0,
        'adl_score': adl_score or 0.0,
        'sport_score': sport_score or 0.0,
        'qol_score': qol_score or 0.0,
        'total_score': total_score or 0.0,
        'is_complete': is_complete
    }

def calculate_ases_scores(response_data: dict) -> dict:
    """
    Calculate ASES scores according to official scoring methodology.
    Pain: (10 - pain_vas) × 5 (0-50 points)
    Function: sum(function_items) × 5/3 (0-50 points)
    Total: Pain + Function (0-100 points, 100 = best)
    """
    # Pain component calculation
    pain_vas = response_data.get('pain_vas', 0)
    pain_component = (10 - pain_vas) * 5
    
    # Function component calculation
    function_items = ['f1_coat', 'f2_sleep', 'f3_wash_back', 'f4_toileting', 'f5_comb_hair', 
                     'f6_high_shelf', 'f7_lift_10lbs', 'f8_throw_ball', 'f9_usual_work', 'f10_usual_sport']
    
    function_values = []
    for item in function_items:
        if item in response_data and response_data[item] is not None:
            function_values.append(response_data[item])
    
    # Check if all function items are answered
    is_complete = len(function_values) == len(function_items) and pain_vas is not None
    
    # Calculate function component
    if function_values:
        function_sum = sum(function_values)
        function_component = function_sum * (5/3)
    else:
        function_component = 0.0
        is_complete = False
    
    # Total score
    total_score = pain_component + function_component
    
    return {
        'pain_component': round(pain_component, 1),
        'function_component': round(function_component, 1),
        'total_score': round(total_score, 1),
        'is_complete': is_complete
    }

# API Routes for Patients
@api_router.post("/patients", response_model=Patient)
async def create_patient(patient: PatientCreate):
    patient_dict = patient.model_dump()
    patient_obj = Patient(**patient_dict)
    patient_data = patient_obj.model_dump()
    
    # Check if email already exists
    existing = await db.patients.find_one({"email": patient.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    result = await db.patients.insert_one(patient_data)
    return patient_obj

@api_router.get("/patients", response_model=List[Patient])
async def get_patients():
    patients = await db.patients.find().to_list(1000)
    return [Patient(**patient) for patient in patients]

@api_router.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return Patient(**patient)

# API Routes for Wearable Data
@api_router.post("/wearable-data", response_model=WearableData)
async def create_wearable_data(data: WearableDataCreate):
    # Verify patient exists
    patient = await db.patients.find_one({"id": data.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    wearable_dict = data.model_dump()
    wearable_obj = WearableData(**wearable_dict)
    wearable_data = wearable_obj.model_dump()
    
    # Check if entry for this date already exists
    existing = await db.wearable_data.find_one({"patient_id": data.patient_id, "date": data.date})
    if existing:
        # Update instead of create
        await db.wearable_data.update_one(
            {"id": existing["id"]},
            {"$set": {**wearable_data, "updated_at": datetime.utcnow()}}
        )
        wearable_data["id"] = existing["id"]
        return WearableData(**wearable_data)
    
    result = await db.wearable_data.insert_one(wearable_data)
    
    # Generate insights after new data
    await generate_insights(data.patient_id)
    
    return wearable_obj

@api_router.get("/wearable-data/{patient_id}", response_model=List[WearableData])
async def get_patient_wearable_data(patient_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    query = {"patient_id": patient_id}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}
    
    data = await db.wearable_data.find(query).sort("date", -1).to_list(1000)
    return [WearableData(**item) for item in data]

# Comprehensive Wearable Data API Routes moved to routers/wearable_api.py
# Legacy endpoints kept for backward compatibility if needed

# API Routes for Surveys
@api_router.post("/surveys", response_model=Survey)
async def create_survey(survey: SurveyCreate):
    # Verify patient exists
    patient = await db.patients.find_one({"id": survey.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    survey_dict = survey.model_dump()
    survey_obj = Survey(**survey_dict)
    survey_data = survey_obj.model_dump()
    
    # Check if survey for this date already exists
    existing = await db.surveys.find_one({"patient_id": survey.patient_id, "date": survey.date})
    if existing:
        # Update instead of create
        await db.surveys.update_one(
            {"id": existing["id"]},
            {"$set": {**survey_data, "updated_at": datetime.utcnow()}}
        )
        survey_data["id"] = existing["id"]
        return Survey(**survey_data)
    
    result = await db.surveys.insert_one(survey_data)
    
    # Generate insights after new data
    await generate_insights(survey.patient_id)
    
    return survey_obj

@api_router.get("/surveys/{patient_id}", response_model=List[Survey])
async def get_patient_surveys(patient_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    query = {"patient_id": patient_id}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}
    
    surveys = await db.surveys.find(query).sort("date", -1).to_list(1000)
    return [Survey(**survey) for survey in surveys]

# API Routes for AI Insights
@api_router.get("/insights/{patient_id}", response_model=List[AIInsight])
async def get_patient_insights(patient_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    query = {"patient_id": patient_id}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}
    
    insights = await db.ai_insights.find(query).sort("date", -1).to_list(1000)
    return [AIInsight(**insight) for insight in insights]

@api_router.post("/generate-insights/{patient_id}", response_model=AIInsight)
async def trigger_insights_generation(patient_id: str):
    insight = await generate_insights(patient_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Could not generate insights. Insufficient data.")
    return insight

# Enhanced AI Insights API Endpoints
@api_router.get("/insights/{patient_id}/detailed", response_model=EnhancedAIInsight)
async def get_detailed_insights(patient_id: str):
    """Get comprehensive AI insights with trajectory analysis"""
    insight = await generate_enhanced_insights(patient_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Could not generate detailed insights. Insufficient data.")
    return insight

@api_router.get("/recovery-trajectory/{patient_id}")
async def get_recovery_trajectory(patient_id: str):
    """Get recovery trajectory data for visualization"""
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    diagnosis_type = DiagnosisType(patient.get("diagnosis_type"))
    body_part = get_body_part(diagnosis_type)
    date_of_surgery = patient.get("date_of_surgery")
    
    if not date_of_surgery:
        raise HTTPException(status_code=400, detail="Surgery date required for trajectory analysis")
    
    weeks_post_surgery = calculate_weeks_post_surgery(date_of_surgery, datetime.utcnow())
    
    # Get trajectory data
    trajectory_data = get_trajectory_for_diagnosis(diagnosis_type)
    if not trajectory_data:
        raise HTTPException(status_code=404, detail="No trajectory data available for this diagnosis")
    
    # Get actual patient scores
    actual_scores = []
    
    if body_part == BodyPart.KNEE:
        # Get all KOOS scores
        koos_scores = await db.koos_scores.find({"patient_id": patient_id}).sort("date", 1).to_list(1000)
        
        for score in koos_scores:
            score_date = score.get("date")
            weeks_at_score = calculate_weeks_post_surgery(date_of_surgery, score_date)
            
            actual_scores.append({
                "week": weeks_at_score,
                "date": score_date,
                "symptoms_score": score.get("symptoms_score", 0),
                "pain_score": score.get("pain_score", 0),
                "adl_score": score.get("adl_score", 0),
                "sport_score": score.get("sport_score", 0),
                "qol_score": score.get("qol_score", 0),
                "total_score": score.get("total_score", 0)
            })
    
    elif body_part == BodyPart.SHOULDER:
        # Get all ASES scores
        ases_scores = await db.ases_scores.find({"patient_id": patient_id}).sort("date", 1).to_list(1000)
        
        for score in ases_scores:
            score_date = score.get("date")
            weeks_at_score = calculate_weeks_post_surgery(date_of_surgery, score_date)
            
            actual_scores.append({
                "week": weeks_at_score,
                "date": score_date,
                "pain_component": score.get("pain_component", 0),
                "function_component": score.get("function_component", 0),
                "total_score": score.get("total_score", 0)
            })
    
    # Format trajectory data for frontend
    formatted_trajectory = {}
    for subscale, points in trajectory_data.items():
        formatted_trajectory[subscale] = [
            {
                "week": point.week,
                "expected_score": point.expected_score,
                "lower_bound": point.lower_bound,
                "upper_bound": point.upper_bound
            }
            for point in points
        ]
    
    return {
        "diagnosis": diagnosis_type.value,
        "body_part": body_part.value,
        "weeks_post_surgery": weeks_post_surgery,
        "trajectory_data": formatted_trajectory,
        "actual_scores": actual_scores,
        "milestones": [
            {
                "week": milestone.week,
                "description": milestone.description,
                "expected_score": milestone.expected_score,
                "critical": milestone.critical,
                "subscale": milestone.subscale
            }
            for milestone in get_milestone_status(diagnosis_type, weeks_post_surgery, {})
        ]
    }

@api_router.get("/risk-assessment/{patient_id}")
async def get_risk_assessment(patient_id: str):
    """Get risk assessment for patient"""
    enhanced_insight = await generate_enhanced_insights(patient_id)
    if not enhanced_insight:
        raise HTTPException(status_code=404, detail="Could not generate risk assessment. Insufficient data.")
    
    return {
        "risk_score": enhanced_insight.risk_assessment.risk_score,
        "risk_category": enhanced_insight.risk_assessment.risk_category,
        "risk_factors": enhanced_insight.risk_assessment.risk_factors,
        "protective_factors": enhanced_insight.risk_assessment.protective_factors,
        "concerning_patterns": enhanced_insight.concerning_patterns,
        "positive_trends": enhanced_insight.positive_trends
    }

@api_router.get("/milestones/{patient_id}")
async def get_recovery_milestones(patient_id: str):
    """Get recovery milestone tracking"""
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    diagnosis_type = DiagnosisType(patient.get("diagnosis_type"))
    date_of_surgery = patient.get("date_of_surgery")
    
    if not date_of_surgery:
        raise HTTPException(status_code=400, detail="Surgery date required for milestone tracking")
    
    weeks_post_surgery = calculate_weeks_post_surgery(date_of_surgery, datetime.utcnow())
    
    # Get current scores
    body_part = get_body_part(diagnosis_type)
    current_scores = {}
    
    if body_part == BodyPart.KNEE:
        latest_koos = await db.koos_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        if latest_koos:
            koos_score = latest_koos[0]
            current_scores = {
                "symptoms_score": koos_score.get("symptoms_score", 0),
                "pain_score": koos_score.get("pain_score", 0),
                "adl_score": koos_score.get("adl_score", 0),
                "sport_score": koos_score.get("sport_score", 0),
                "qol_score": koos_score.get("qol_score", 0),
                "total_score": koos_score.get("total_score", 0)
            }
    
    elif body_part == BodyPart.SHOULDER:
        latest_ases = await db.ases_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        if latest_ases:
            ases_score = latest_ases[0]
            current_scores = {
                "pain_component": ases_score.get("pain_component", 0),
                "function_component": ases_score.get("function_component", 0),
                "total_score": ases_score.get("total_score", 0)
            }
    
    # Get milestone status
    milestone_status = get_milestone_status(diagnosis_type, weeks_post_surgery, current_scores)
    
    return {
        "diagnosis": diagnosis_type.value,
        "weeks_post_surgery": weeks_post_surgery,
        "milestones": milestone_status,
        "critical_milestones_missed": len([m for m in milestone_status if m["critical"] and not m["achieved"]]),
        "total_milestones_achieved": len([m for m in milestone_status if m["achieved"]])
    }

# Recovery indicators analysis
async def analyze_recovery_indicators(patient_id: str) -> Dict[str, Any]:
    """Analyze comprehensive wearable data for recovery indicators and alerts"""
    from models.wearable_data import RecoveryIndicators
    
    # Get recent comprehensive wearable data
    recent_data = await db.comprehensive_wearable_data.find({
        "patient_id": patient_id
    }).sort("date", -1).limit(14).to_list(14)
    
    if not recent_data:
        return {"error": "No wearable data available"}
    
    indicators = RecoveryIndicators(
        patient_id=patient_id,
        analysis_date=datetime.utcnow().date()
    )
    
    # Analyze walking speed trends
    walking_speeds = [
        d.get("movement_metrics", {}).get("walking_speed_ms") 
        for d in recent_data 
        if d.get("movement_metrics", {}).get("walking_speed_ms") is not None
    ]
    
    if len(walking_speeds) >= 3:
        recent_speed = walking_speeds[0]
        older_speed = sum(walking_speeds[-3:]) / 3
        if recent_speed < older_speed * 0.9:  # 10% decline
            indicators.walking_speed_trend = "declining"
        elif recent_speed > older_speed * 1.1:  # 10% improvement
            indicators.walking_speed_trend = "improving"
        else:
            indicators.walking_speed_trend = "stable"
    
    # Analyze activity consistency
    step_counts = [
        d.get("activity_metrics", {}).get("steps") 
        for d in recent_data 
        if d.get("activity_metrics", {}).get("steps") is not None
    ]
    
    if len(step_counts) >= 7:
        avg_steps = sum(step_counts) / len(step_counts)
        step_variance = sum((s - avg_steps) ** 2 for s in step_counts) / len(step_counts)
        step_std = step_variance ** 0.5
        indicators.activity_consistency = max(0, 1 - (step_std / avg_steps)) if avg_steps > 0 else 0
        
        # Check for significant activity drops
        recent_avg = sum(step_counts[:3]) / 3 if len(step_counts) >= 3 else step_counts[0]
        baseline_avg = sum(step_counts[-7:]) / 7
        if recent_avg < baseline_avg * 0.7:  # 30% drop
            indicators.activity_drop_alert = True
    
    # Analyze sleep quality trends
    sleep_efficiencies = [
        d.get("sleep_metrics", {}).get("sleep_efficiency") 
        for d in recent_data 
        if d.get("sleep_metrics", {}).get("sleep_efficiency") is not None
    ]
    
    if len(sleep_efficiencies) >= 3:
        recent_sleep = sleep_efficiencies[0]
        older_sleep = sum(sleep_efficiencies[-3:]) / 3
        if recent_sleep < older_sleep * 0.9:
            indicators.sleep_quality_trend = "declining"
        elif recent_sleep > older_sleep * 1.1:
            indicators.sleep_quality_trend = "improving"
        else:
            indicators.sleep_quality_trend = "stable"
            
        indicators.sleep_efficiency_avg = sum(sleep_efficiencies) / len(sleep_efficiencies)
        
        # Check for sleep disruption
        if indicators.sleep_efficiency_avg < 75:  # Below 75% efficiency
            indicators.sleep_disruption_alert = True
    
    # Check sedentary time alerts
    sedentary_times = [
        d.get("activity_metrics", {}).get("sedentary_minutes") 
        for d in recent_data 
        if d.get("activity_metrics", {}).get("sedentary_minutes") is not None
    ]
    
    if sedentary_times:
        avg_sedentary = sum(sedentary_times) / len(sedentary_times)
        if avg_sedentary > 600:  # More than 10 hours sedentary
            indicators.sedentary_time_alert = True
    
    # Store the analysis
    indicators_data = indicators.model_dump()
    await db.recovery_indicators.insert_one(indicators_data)
    
    return indicators_data

def calculate_simple_correlation(x: List[float], y: List[float]) -> float:
    """Calculate simple correlation coefficient"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

# Enhanced AI insights generation
async def generate_enhanced_insights(patient_id: str) -> Optional[EnhancedAIInsight]:
    """Generate enhanced AI insights with trajectory analysis and predictive analytics"""
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        return None
    
    diagnosis_type = DiagnosisType(patient.get("diagnosis_type"))
    body_part = get_body_part(diagnosis_type)
    date_of_surgery = patient.get("date_of_surgery")
    insight_date = datetime.utcnow()
    
    # Calculate weeks post-surgery
    weeks_post_surgery = calculate_weeks_post_surgery(date_of_surgery, insight_date) if date_of_surgery else 0
    
    # Initialize analytics engines
    try:
        from services.wearable_analytics import WearableAnalyticsEngine
        from services.recovery_correlation_engine import RecoveryCorrelationEngine
        from services.clinical_alerts import ClinicalAlertsEngine
        from services.predictive_modeling import EnhancedPredictiveModeling
        from utils.recovery_metrics import RecoveryMetricsCalculator
        
        analytics_engine = WearableAnalyticsEngine(db)
        correlation_engine = RecoveryCorrelationEngine(db)
        alerts_engine = ClinicalAlertsEngine(db)
        predictive_engine = EnhancedPredictiveModeling(db)
    except Exception as e:
        logger.warning(f"Could not initialize analytics engines: {e}")
        analytics_engine = correlation_engine = alerts_engine = predictive_engine = None
    
    # Get current and previous scores
    current_scores = {}
    previous_scores = {}
    
    if body_part == BodyPart.KNEE:
        # Get latest KOOS scores
        latest_koos = await db.koos_scores.find({"patient_id": patient_id}).sort("date", -1).limit(2).to_list(2)
        
        if latest_koos:
            current_koos = latest_koos[0]
            insight_date = current_koos.get("date", datetime.utcnow())
            
            current_scores = {
                "symptoms_score": current_koos.get("symptoms_score", 0),
                "pain_score": current_koos.get("pain_score", 0),
                "adl_score": current_koos.get("adl_score", 0),
                "sport_score": current_koos.get("sport_score", 0),
                "qol_score": current_koos.get("qol_score", 0),
                "total_score": current_koos.get("total_score", 0)
            }
            
            # Get previous scores for velocity calculation
            if len(latest_koos) > 1:
                previous_koos = latest_koos[1]
                previous_scores = {
                    "symptoms_score": previous_koos.get("symptoms_score", 0),
                    "pain_score": previous_koos.get("pain_score", 0),
                    "adl_score": previous_koos.get("adl_score", 0),
                    "sport_score": previous_koos.get("sport_score", 0),
                    "qol_score": previous_koos.get("qol_score", 0),
                    "total_score": previous_koos.get("total_score", 0)
                }
    
    elif body_part == BodyPart.SHOULDER:
        # Get latest ASES scores
        latest_ases = await db.ases_scores.find({"patient_id": patient_id}).sort("date", -1).limit(2).to_list(2)
        
        if latest_ases:
            current_ases = latest_ases[0]
            insight_date = current_ases.get("date", datetime.utcnow())
            
            current_scores = {
                "pain_component": current_ases.get("pain_component", 0),
                "function_component": current_ases.get("function_component", 0),
                "total_score": current_ases.get("total_score", 0)
            }
            
            # Get previous scores for velocity calculation
            if len(latest_ases) > 1:
                previous_ases = latest_ases[1]
                previous_scores = {
                    "pain_component": previous_ases.get("pain_component", 0),
                    "function_component": previous_ases.get("function_component", 0),
                    "total_score": previous_ases.get("total_score", 0)
                }
    
    # If no PRO scores available, return None
    if not current_scores:
        return None
    
    # Trajectory analysis
    trajectory_analysis = []
    for subscale, actual_score in current_scores.items():
        expected_point = get_expected_score_at_week(diagnosis_type, weeks_post_surgery, subscale)
        if expected_point:
            status = get_recovery_status_from_trajectory(actual_score, expected_point)
            deviation = ((actual_score - expected_point.expected_score) / expected_point.expected_score) * 100
            
            trajectory_analysis.append(TrajectoryAnalysis(
                subscale=subscale,
                actual_score=actual_score,
                expected_score=expected_point.expected_score,
                lower_bound=expected_point.lower_bound,
                upper_bound=expected_point.upper_bound,
                status=status,
                deviation_percentage=deviation
            ))
    
    # Milestone status
    milestone_status_data = get_milestone_status(diagnosis_type, weeks_post_surgery, current_scores)
    milestone_status = [
        RecoveryMilestoneStatus(**milestone) for milestone in milestone_status_data
    ]
    
    # Recovery velocity
    recovery_velocity = RecoveryPredictor.calculate_recovery_velocity(
        current_scores, previous_scores, weeks_between=2
    )
    velocity_objects = [RecoveryVelocity(**vel) for vel in recovery_velocity]
    
    # Get comprehensive wearable data
    wearable_data = None
    comprehensive_wearable = await db.comprehensive_wearable_data.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    if comprehensive_wearable:
        wearable_data = comprehensive_wearable[0]
    else:
        # Fall back to legacy wearable data if comprehensive data not available
        latest_wearable = await db.wearable_data.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        if latest_wearable:
            wearable_data = latest_wearable[0]
    
    # Enhanced analytics integration
    enhanced_analytics = {}
    if analytics_engine:
        try:
            # Get recovery velocity analysis
            velocity_analysis = await analytics_engine.analyze_recovery_velocity(patient_id)
            if "error" not in velocity_analysis:
                enhanced_analytics["velocity_analysis"] = velocity_analysis
            
            # Get clinical risk assessment
            clinical_risk = await analytics_engine.assess_clinical_risk(patient_id)
            if "error" not in clinical_risk:
                enhanced_analytics["clinical_risk"] = clinical_risk
            
            # Get personalized insights
            personalized_insights = await analytics_engine.generate_personalized_insights(patient_id)
            if "error" not in personalized_insights:
                enhanced_analytics["personalized_insights"] = personalized_insights
                
        except Exception as e:
            logger.warning(f"Enhanced analytics failed: {e}")
    
    # Correlation analysis integration
    correlation_insights = {}
    if correlation_engine:
        try:
            # Get comprehensive correlations
            correlations = await correlation_engine.analyze_comprehensive_correlations(patient_id)
            if "error" not in correlations:
                correlation_insights["comprehensive"] = correlations
                
            # Get activity-pain correlation
            activity_pain = await correlation_engine.analyze_activity_pain_correlation(patient_id)
            if "error" not in activity_pain:
                correlation_insights["activity_pain"] = activity_pain
                
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
    
    # Predictive modeling integration
    predictive_insights = {}
    if predictive_engine:
        try:
            # Get recovery timeline prediction
            timeline_prediction = await predictive_engine.predict_recovery_timeline(patient_id)
            if "error" not in timeline_prediction:
                predictive_insights["timeline"] = timeline_prediction
                
            # Get complication risk prediction
            risk_prediction = await predictive_engine.predict_complication_risk(patient_id)
            if "error" not in risk_prediction:
                predictive_insights["complication_risk"] = risk_prediction
                
        except Exception as e:
            logger.warning(f"Predictive modeling failed: {e}")
    
    # Clinical alerts integration
    clinical_alerts = []
    if alerts_engine:
        try:
            alerts = await alerts_engine.generate_real_time_alerts(patient_id)
            clinical_alerts = [
                {
                    "severity": alert.severity.value,
                    "category": alert.alert_type.value,
                    "title": alert.title,
                    "description": alert.description,
                    "recommendations": alert.recommendations
                }
                for alert in alerts
            ]
        except Exception as e:
            logger.warning(f"Clinical alerts generation failed: {e}")
    
    # Risk assessment (enhanced with new analytics)
    if enhanced_analytics.get("clinical_risk"):
        risk_score = enhanced_analytics["clinical_risk"].get("overall_risk_score", 50)
        risk_category = enhanced_analytics["clinical_risk"].get("risk_category", "moderate")
    else:
        # Fall back to original risk calculation
        risk_score = RecoveryPredictor.calculate_risk_score(
            diagnosis_type, weeks_post_surgery, current_scores, previous_scores, wearable_data
        )
        risk_category = RecoveryPredictor.get_risk_category(risk_score)
    
    # Identify concerning patterns and positive trends (enhanced)
    if enhanced_analytics.get("clinical_risk"):
        concerning_patterns = enhanced_analytics["clinical_risk"].get("detailed_assessment", {}).get("risk_factors", [])
        positive_trends = enhanced_analytics["clinical_risk"].get("detailed_assessment", {}).get("protective_factors", [])
    else:
        # Fall back to original pattern detection
        concerning_patterns = RecoveryPredictor.detect_concerning_patterns(
            diagnosis_type, weeks_post_surgery, current_scores, recovery_velocity, milestone_status_data
        )
        
        positive_trends = RecoveryPredictor.identify_positive_trends(
            diagnosis_type, weeks_post_surgery, current_scores, recovery_velocity, milestone_status_data
        )
    
    # Generate recommendations (enhanced with new analytics)
    trajectory_analysis_dict = [t.model_dump() for t in trajectory_analysis]
    
    # Combine original recommendations with new analytics recommendations
    original_recommendations = RecommendationEngine.generate_recommendations(
        diagnosis_type, weeks_post_surgery, current_scores, 
        trajectory_analysis_dict, milestone_status_data, concerning_patterns, risk_score
    )
    
    # Add recommendations from enhanced analytics
    enhanced_recommendations = []
    if enhanced_analytics.get("personalized_insights"):
        personalized = enhanced_analytics["personalized_insights"].get("personalized_insights", {})
        for insight_type, insight_data in personalized.items():
            if isinstance(insight_data, dict) and "recommendations" in insight_data:
                enhanced_recommendations.extend(insight_data["recommendations"])
    
    # Add recommendations from clinical alerts
    for alert in clinical_alerts:
        enhanced_recommendations.extend(alert.get("recommendations", []))
    
    # Combine all recommendations
    all_recommendations = original_recommendations + [
        {"text": rec, "priority": "medium", "category": "enhanced_analytics"}
        for rec in enhanced_recommendations[:5]  # Limit to top 5
    ]
    
    recommendation_objects = [RecommendationItem(**rec) for rec in all_recommendations]
    
    # Predict recovery timeline (enhanced with new predictive modeling)
    if predictive_insights.get("timeline"):
        timeline_data = predictive_insights["timeline"]
        projected_recovery_date = timeline_data.get("overall_timeline", {}).get("full_recovery_date")
        confidence_interval = f"±{timeline_data.get('overall_timeline', {}).get('confidence', 0.7)*100:.0f}%"
    else:
        # Fall back to original prediction
        projected_recovery_date, confidence_interval = RecoveryPredictor.predict_recovery_timeline(
            diagnosis_type, weeks_post_surgery, current_scores, recovery_velocity
        )
    
    # Determine overall recovery status
    avg_trajectory_status = [t.status for t in trajectory_analysis]
    if "Behind Schedule" in avg_trajectory_status:
        recovery_status = RecoveryStatus.NEEDS_ATTENTION
    elif "Slightly Behind" in avg_trajectory_status:
        recovery_status = RecoveryStatus.AT_RISK
    else:
        recovery_status = RecoveryStatus.ON_TRACK
    
    # Generate key findings (enhanced with new analytics)
    key_findings = []
    
    # Add trajectory-based findings
    behind_schedule = [t for t in trajectory_analysis if "Behind" in t.status]
    if behind_schedule:
        key_findings.append(f"{len(behind_schedule)} recovery domains below expected trajectory")
    
    # Add milestone findings
    missed_critical = [m for m in milestone_status if m.critical and not m.achieved]
    if missed_critical:
        key_findings.append(f"Critical milestone missed: {missed_critical[0].description}")
    
    # Add velocity findings
    declining_areas = [v for v in recovery_velocity if v["trend"] == "Declining"]
    if declining_areas:
        key_findings.append(f"Declining progress in {len(declining_areas)} areas")
    
    # Add enhanced analytics findings
    if enhanced_analytics.get("velocity_analysis"):
        velocity_data = enhanced_analytics["velocity_analysis"]
        for metric, data in velocity_data.get("velocity_metrics", {}).items():
            if isinstance(data, dict) and data.get("trend") == "declining":
                key_findings.append(f"Declining {metric} velocity detected")
    
    # Add correlation insights
    if correlation_insights.get("comprehensive"):
        corr_insights = correlation_insights["comprehensive"].get("insights", [])
        key_findings.extend(corr_insights[:2])  # Add top 2 correlation insights
    
    # Add clinical alerts as findings
    critical_alerts = [alert for alert in clinical_alerts if alert.get("severity") == "critical"]
    if critical_alerts:
        key_findings.append(f"{len(critical_alerts)} critical clinical alerts active")
    
    # Add predictive insights
    if predictive_insights.get("complication_risk"):
        risk_data = predictive_insights["complication_risk"]
        overall_risk = risk_data.get("overall_risk", {})
        if overall_risk.get("risk_category") in ["high", "very_high"]:
            key_findings.append(f"Elevated complication risk: {overall_risk.get('risk_category')}")
    
    # Calculate progress percentage
    progress_percentage = sum(current_scores.values()) / len(current_scores)
    
    # Create risk assessment
    risk_assessment = RiskAssessment(
        risk_score=risk_score,
        risk_category=risk_category,
        risk_factors=concerning_patterns,
        protective_factors=positive_trends
    )
    
    # Create enhanced insight
    enhanced_insight = EnhancedAIInsight(
        patient_id=patient_id,
        date=insight_date,
        weeks_post_surgery=weeks_post_surgery,
        recovery_status=recovery_status,
        recommendations=recommendation_objects,
        progress_percentage=progress_percentage,
        trajectory_analysis=trajectory_analysis,
        milestone_status=milestone_status,
        risk_assessment=risk_assessment,
        recovery_velocity=velocity_objects,
        projected_recovery_date=projected_recovery_date,
        confidence_interval=confidence_interval,
        key_findings=key_findings,
        concerning_patterns=concerning_patterns,
        positive_trends=positive_trends
    )
    
    # Save to database with enhanced analytics data
    insight_data = enhanced_insight.model_dump()
    
    # Add enhanced analytics data to insight
    insight_data["enhanced_analytics"] = enhanced_analytics
    insight_data["correlation_insights"] = correlation_insights
    insight_data["predictive_insights"] = predictive_insights
    insight_data["clinical_alerts"] = clinical_alerts
    insight_data["analytics_timestamp"] = datetime.utcnow()
    
    # Check if insight for this date already exists
    existing = await db.enhanced_ai_insights.find_one({
        "patient_id": patient_id,
        "date": insight_date
    })
    
    if existing:
        # Update instead of create
        await db.enhanced_ai_insights.update_one(
            {"id": existing["id"]},
            {"$set": {**insight_data, "updated_at": datetime.utcnow()}}
        )
        insight_data["id"] = existing["id"]
        return EnhancedAIInsight(**insight_data)
    
    result = await db.enhanced_ai_insights.insert_one(insight_data)
    return enhanced_insight

# Helper function to generate AI insights
async def generate_insights(patient_id: str) -> Optional[AIInsight]:
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        return None
    
    diagnosis_type = DiagnosisType(patient.get("diagnosis_type"))
    body_part = get_body_part(diagnosis_type)
    recovery_status = RecoveryStatus.ON_TRACK
    recommendations = []
    risk_factors = []
    progress_percentage = 0.0
    insight_date = datetime.utcnow()
    
    # Try to get PRO scores first, fall back to basic survey if needed
    if body_part == BodyPart.KNEE:
        # Get latest KOOS scores
        latest_koos = await db.koos_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        
        if latest_koos:
            koos_score = latest_koos[0]
            insight_date = koos_score.get("date", datetime.utcnow())
            
            # Enhanced AI logic using KOOS subscales
            symptoms_score = koos_score.get("symptoms_score", 0)
            pain_score = koos_score.get("pain_score", 0)
            adl_score = koos_score.get("adl_score", 0)
            sport_score = koos_score.get("sport_score", 0)
            qol_score = koos_score.get("qol_score", 0)
            total_score = koos_score.get("total_score", 0)
            
            # Determine recovery status based on KOOS scores
            if total_score < 40:
                recovery_status = RecoveryStatus.NEEDS_ATTENTION
                recommendations.append("Consider consulting with your physician about current recovery progress")
            elif total_score < 60:
                recovery_status = RecoveryStatus.AT_RISK
                recommendations.append("Focus on prescribed rehabilitation exercises")
            
            # Specific subscale analysis
            if pain_score < 50:
                risk_factors.append("Significant knee pain reported")
                recommendations.append("Pain management strategies may be beneficial")
            
            if symptoms_score < 60:
                risk_factors.append("Knee symptoms affecting daily activities")
                recommendations.append("Monitor for swelling, stiffness, and mechanical symptoms")
            
            if adl_score < 70:
                risk_factors.append("Difficulty with activities of daily living")
                recommendations.append("Focus on functional movement exercises")
            
            if sport_score < 50:
                risk_factors.append("Limited sports/recreation capability")
                recommendations.append("Progress gradually toward sports-specific activities")
            
            if qol_score < 60:
                risk_factors.append("Reduced quality of life due to knee problems")
                recommendations.append("Consider psychological support for recovery confidence")
            
            # Calculate progress percentage from total KOOS score
            progress_percentage = min(max(total_score, 0), 100)
            
        else:
            # Fallback to basic survey for knee patients
            latest_survey = await db.surveys.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
            if latest_survey:
                survey = latest_survey[0]
                insight_date = survey.get("date", datetime.utcnow())
                pain_score = survey.get("pain_score", 0)
                mobility_score = survey.get("mobility_score", 0)
                
                # Convert 0-10 scores to percentage for consistency
                progress_percentage = ((10 - pain_score) * 0.4 + mobility_score * 0.6) * 10
                
                if pain_score > 7:
                    recovery_status = RecoveryStatus.NEEDS_ATTENTION
                    risk_factors.append("High pain levels")
                    recommendations.append("Consider KOOS questionnaire for detailed assessment")
    
    elif body_part == BodyPart.SHOULDER:
        # Get latest ASES scores
        latest_ases = await db.ases_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        
        if latest_ases:
            ases_score = latest_ases[0]
            insight_date = ases_score.get("date", datetime.utcnow())
            
            # Enhanced AI logic using ASES components
            pain_component = ases_score.get("pain_component", 0)
            function_component = ases_score.get("function_component", 0)
            total_score = ases_score.get("total_score", 0)
            
            # Determine recovery status based on ASES scores
            if total_score < 40:
                recovery_status = RecoveryStatus.NEEDS_ATTENTION
                recommendations.append("Consider consulting with your physician about current recovery progress")
            elif total_score < 60:
                recovery_status = RecoveryStatus.AT_RISK
                recommendations.append("Focus on prescribed rehabilitation exercises")
            
            # Component-specific analysis
            if pain_component < 25:  # Less than half of pain component
                risk_factors.append("Significant shoulder pain reported")
                recommendations.append("Pain management and activity modification may be needed")
            
            if function_component < 25:  # Less than half of function component
                risk_factors.append("Limited shoulder function affecting daily activities")
                recommendations.append("Focus on range of motion and strengthening exercises")
            
            # Calculate progress percentage from total ASES score
            progress_percentage = min(max(total_score, 0), 100)
            
        else:
            # Fallback to basic survey for shoulder patients
            latest_survey = await db.surveys.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
            if latest_survey:
                survey = latest_survey[0]
                insight_date = survey.get("date", datetime.utcnow())
                pain_score = survey.get("pain_score", 0)
                mobility_score = survey.get("mobility_score", 0)
                
                # Convert 0-10 scores to percentage for consistency
                progress_percentage = ((10 - pain_score) * 0.4 + mobility_score * 0.6) * 10
                
                if pain_score > 7:
                    recovery_status = RecoveryStatus.NEEDS_ATTENTION
                    risk_factors.append("High pain levels")
                    recommendations.append("Consider ASES questionnaire for detailed assessment")
    
    # Add wearable data insights regardless of injury type
    latest_wearable = await db.comprehensive_wearable_data.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    if latest_wearable:
        wearable = latest_wearable[0]
        
        # Extract metrics from comprehensive data structure
        steps = wearable.get("activity_metrics", {}).get("steps", 0) if wearable.get("activity_metrics") else 0
        sleep_hours = None
        sleep_metrics = wearable.get("sleep_metrics")
        if sleep_metrics and sleep_metrics.get("total_sleep_minutes"):
            sleep_hours = sleep_metrics.get("total_sleep_minutes") / 60
    else:
        # Fall back to legacy wearable data
        legacy_wearable = await db.wearable_data.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
        if legacy_wearable:
            wearable = legacy_wearable[0]
            steps = wearable.get("steps", 0)
            sleep_hours = wearable.get("sleep_hours", 0)
        else:
            wearable = None
            steps = 0
            sleep_hours = 0
    
    if wearable:
        if steps < 2000:
            if recovery_status != RecoveryStatus.NEEDS_ATTENTION:
                recovery_status = RecoveryStatus.AT_RISK
            risk_factors.append("Very low activity level")
            recommendations.append("Gradually increase daily activity as tolerated")
        
        if sleep_hours and sleep_hours < 6:
            risk_factors.append("Insufficient sleep")
            recommendations.append("Adequate sleep is crucial for tissue healing")
        
        # Add comprehensive wearable data insights
        if wearable.get("heart_rate_metrics"):
            hr_metrics = wearable.get("heart_rate_metrics")
            if hr_metrics.get("resting_hr") and hr_metrics.get("resting_hr") > 100:
                risk_factors.append("Elevated resting heart rate may indicate stress or poor recovery")
                recommendations.append("Monitor stress levels and ensure adequate rest")
        
        if wearable.get("movement_metrics"):
            movement = wearable.get("movement_metrics")
            if movement.get("walking_speed_ms") and movement.get("walking_speed_ms") < 1.0:  # Very slow walking
                risk_factors.append("Significantly reduced walking speed")
                recommendations.append("Focus on gait training and mobility exercises")
            
            if movement.get("mobility_score") and movement.get("mobility_score") < 50:
                risk_factors.append("Low mobility score indicates movement limitations")
                recommendations.append("Consider physical therapy evaluation")
        
        if wearable.get("sleep_metrics"):
            sleep = wearable.get("sleep_metrics")
            if sleep.get("sleep_efficiency") and sleep.get("sleep_efficiency") < 70:
                risk_factors.append("Poor sleep efficiency affecting recovery")
                recommendations.append("Address sleep hygiene and pain management")
        
        if wearable.get("activity_metrics"):
            activity = wearable.get("activity_metrics")
            if activity.get("sedentary_minutes") and activity.get("sedentary_minutes") > 600:  # > 10 hours
                risk_factors.append("Excessive sedentary time")
                recommendations.append("Incorporate regular movement breaks throughout the day")
    
    # Ensure we have some data to generate insights
    if not risk_factors and not recommendations:
        recommendations.append("Continue with current rehabilitation program")
        recommendations.append("Complete validated questionnaires regularly for better tracking")
    
    # Create insight object
    insight = AIInsight(
        patient_id=patient_id,
        date=insight_date,
        recovery_status=recovery_status,
        recommendations=recommendations,
        risk_factors=risk_factors,
        progress_percentage=progress_percentage
    )
    
    # Save to database
    insight_data = insight.model_dump()
    
    # Check if insight for this date already exists
    existing = await db.ai_insights.find_one({
        "patient_id": patient_id,
        "date": insight_date
    })
    
    if existing:
        # Update instead of create
        await db.ai_insights.update_one(
            {"id": existing["id"]},
            {"$set": {**insight_data, "updated_at": datetime.utcnow()}}
        )
        insight_data["id"] = existing["id"]
        return AIInsight(**insight_data)
    
    result = await db.ai_insights.insert_one(insight_data)
    return insight

# Add sample data route for demonstration
@api_router.post("/sample-data")
async def add_sample_data():
    # Sample patients with varied diagnoses
    patients = [
        PatientCreate(
            name="John Smith",
            email="john.smith@example.com",
            diagnosis_type=DiagnosisType.ACL_TEAR,
            date_of_injury=datetime.strptime("2025-01-15", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-01", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Sarah Johnson",
            email="sarah.johnson@example.com",
            diagnosis_type=DiagnosisType.ROTATOR_CUFF_TEAR,
            date_of_injury=datetime.strptime("2025-02-05", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-20", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Michael Brown",
            email="michael.brown@example.com",
            diagnosis_type=DiagnosisType.MENISCUS_TEAR,
            date_of_injury=datetime.strptime("2025-01-10", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-01-25", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Emma Wilson",
            email="emma.wilson@example.com",
            diagnosis_type=DiagnosisType.LABRAL_TEAR,
            date_of_injury=datetime.strptime("2025-01-20", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-10", "%Y-%m-%d")
        ),
        PatientCreate(
            name="David Chen",
            email="david.chen@example.com",
            diagnosis_type=DiagnosisType.KNEE_OSTEOARTHRITIS,
            date_of_injury=datetime.strptime("2024-12-15", "%Y-%m-%d"),
            date_of_surgery=None
        ),
        PatientCreate(
            name="Lisa Martinez",
            email="lisa.martinez@example.com",
            diagnosis_type=DiagnosisType.SHOULDER_INSTABILITY,
            date_of_injury=datetime.strptime("2025-01-05", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-01-30", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Robert Taylor",
            email="robert.taylor@example.com",
            diagnosis_type=DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT,
            date_of_injury=datetime.strptime("2024-11-20", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2024-12-15", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Jennifer Lee",
            email="jennifer.lee@example.com",
            diagnosis_type=DiagnosisType.CARTILAGE_DEFECT,
            date_of_injury=datetime.strptime("2025-01-08", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-05", "%Y-%m-%d")
        )
    ]
    
    # Add patients
    created_patients = []
    for p in patients:
        # Check if patient already exists
        existing = await db.patients.find_one({"email": p.email})
        if existing:
            created_patients.append(Patient(**existing))
            continue
            
        patient = Patient(**p.model_dump())
        await db.patients.insert_one(patient.model_dump())
        created_patients.append(patient)
    
    # Add wearable data for each patient
    for i, patient in enumerate(created_patients):
        # Generate data for last 10 days
        today = datetime.now().date()
        for days_ago in range(10, 0, -1):
            sample_date = datetime.now() - timedelta(days=days_ago)
            
            # Different data patterns based on recovery progress
            base_steps = 2000 + (10 - days_ago) * 300  # Steps increase over time
            
            # Vary by patient condition based on body part
            patient_body_part = get_body_part(patient.diagnosis_type)
            if patient_body_part == BodyPart.KNEE:
                # Knee patients might have more restricted mobility initially
                step_multiplier = 0.7
                heart_rate_base = 75
            else:
                # Shoulder patients might have less impact on walking
                step_multiplier = 0.9
                heart_rate_base = 72
            
            wearable_data = WearableDataCreate(
                patient_id=patient.id,
                date=sample_date,
                steps=int(base_steps * step_multiplier * (0.9 + 0.2 * (i % 3))),  # Vary by patient
                heart_rate=heart_rate_base + (days_ago % 5),
                oxygen_saturation=95 + (days_ago % 3),
                sleep_hours=6.5 + (days_ago % 3) * 0.5,
                walking_speed=2.0 + (10 - days_ago) * 0.1
            )
            
            # Check if data already exists
            existing = await db.wearable_data.find_one({
                "patient_id": patient.id,
                "date": sample_date
            })
            
            if not existing:
                wearable_obj = WearableData(**wearable_data.model_dump())
                await db.wearable_data.insert_one(wearable_obj.model_dump())
        
        # Add survey data
        for days_ago in range(10, 0, -1):
            sample_date = datetime.now() - timedelta(days=days_ago)
            
            # Pain decreases over time
            base_pain = max(8 - (10 - days_ago) * 0.5, 2)
            # Mobility increases over time
            base_mobility = min(3 + (10 - days_ago) * 0.5, 8)
            
            # Adjust based on body part
            if patient_body_part == BodyPart.KNEE:
                # Knee specific values
                rom_fields = {
                    "knee_flexion": 70 + (10 - days_ago) * 2,
                    "knee_extension": -15 + (10 - days_ago) * 1
                }
                strength_fields = {
                    "quadriceps": 3 + min((10 - days_ago) // 3, 2),
                    "hamstrings": 2 + min((10 - days_ago) // 4, 2)
                }
                adl_fields = {
                    "walking": 3 + min((10 - days_ago) // 3, 4),
                    "stairs": 1 + min((10 - days_ago) // 4, 5),
                    "standing_from_chair": 2 + min((10 - days_ago) // 3, 4)
                }
            else:
                # Shoulder specific values
                rom_fields = {
                    "shoulder_flexion": 90 + (10 - days_ago) * 3,
                    "shoulder_abduction": 70 + (10 - days_ago) * 2.5,
                    "external_rotation": 15 + (10 - days_ago) * 2
                }
                strength_fields = {
                    "deltoid": 2 + min((10 - days_ago) // 3, 3),
                    "rotator_cuff": 1 + min((10 - days_ago) // 4, 4)
                }
                adl_fields = {
                    "reaching_overhead": 1 + min((10 - days_ago) // 3, 6),
                    "carrying_objects": 2 + min((10 - days_ago) // 4, 5),
                    "dressing": 3 + min((10 - days_ago) // 3, 4)
                }
            
            survey_data = SurveyCreate(
                patient_id=patient.id,
                date=sample_date,
                pain_score=int(base_pain * (0.9 + 0.2 * (i % 3))),  # Vary by patient
                mobility_score=int(base_mobility * (0.9 + 0.2 * (i % 3))),  # Vary by patient
                range_of_motion=rom_fields,
                strength=strength_fields,
                activities_of_daily_living=adl_fields,
                notes=f"Patient reported {'improved' if days_ago < 5 else 'some'} comfort today."
            )
            
            # Check if survey already exists
            existing = await db.surveys.find_one({
                "patient_id": patient.id,
                "date": sample_date
            })
            
            if not existing:
                survey_obj = Survey(**survey_data.model_dump())
                await db.surveys.insert_one(survey_obj.model_dump())
            
            # Generate insights for each day
            await generate_insights(patient.id)
    
    return {"message": "Sample data added successfully"}

# API Routes for KOOS
@api_router.post("/koos", response_model=KOOSResponse)
async def create_koos_response(koos_data: KOOSCreate):
    # Verify patient exists and has ACL injury
    patient = await db.patients.find_one({"id": koos_data.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient_body_part = get_body_part(DiagnosisType(patient.get("diagnosis_type")))
    if patient_body_part != BodyPart.KNEE:
        raise HTTPException(status_code=400, detail="KOOS questionnaire is only valid for knee injuries")
    
    # Create KOOS response
    koos_dict = koos_data.model_dump()
    koos_obj = KOOSResponse(**koos_dict)
    koos_response_data = koos_obj.model_dump()
    
    # Check if response for this date already exists
    existing = await db.koos_responses.find_one({"patient_id": koos_data.patient_id, "date": koos_data.date})
    if existing:
        # Update existing response
        await db.koos_responses.update_one(
            {"id": existing["id"]},
            {"$set": {**koos_response_data, "updated_at": datetime.utcnow()}}
        )
        koos_response_data["id"] = existing["id"]
        koos_obj = KOOSResponse(**koos_response_data)
    else:
        # Insert new response
        await db.koos_responses.insert_one(koos_response_data)
    
    # Calculate scores
    scores_data = calculate_koos_scores(koos_dict)
    scores_obj = KOOSScores(
        patient_id=koos_data.patient_id,
        date=koos_data.date,
        koos_response_id=koos_obj.id,
        **scores_data
    )
    scores_data_full = scores_obj.model_dump()
    
    # Check if scores for this date already exist
    existing_scores = await db.koos_scores.find_one({"patient_id": koos_data.patient_id, "date": koos_data.date})
    if existing_scores:
        # Update existing scores
        await db.koos_scores.update_one(
            {"id": existing_scores["id"]},
            {"$set": {**scores_data_full, "updated_at": datetime.utcnow()}}
        )
    else:
        # Insert new scores
        await db.koos_scores.insert_one(scores_data_full)
    
    # Generate updated insights
    await generate_insights(koos_data.patient_id)
    
    return koos_obj

@api_router.get("/koos/{patient_id}", response_model=List[KOOSScores])
async def get_patient_koos_scores(patient_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    query = {"patient_id": patient_id}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}
    
    scores = await db.koos_scores.find(query).sort("date", -1).to_list(1000)
    return [KOOSScores(**score) for score in scores]

@api_router.get("/koos/{patient_id}/latest", response_model=Optional[KOOSScores])
async def get_latest_koos_scores(patient_id: str):
    latest_score = await db.koos_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    if latest_score:
        return KOOSScores(**latest_score[0])
    return None

@api_router.get("/koos/{patient_id}/trends")
async def get_koos_trends(patient_id: str, days: int = 90):
    start_date = datetime.utcnow() - timedelta(days=days)
    scores = await db.koos_scores.find({
        "patient_id": patient_id,
        "date": {"$gte": start_date}
    }).sort("date", 1).to_list(1000)
    
    trends = []
    for score in scores:
        trends.append({
            "date": score["date"],
            "symptoms_score": score["symptoms_score"],
            "pain_score": score["pain_score"],
            "adl_score": score["adl_score"],
            "sport_score": score["sport_score"],
            "qol_score": score["qol_score"],
            "total_score": score["total_score"]
        })
    
    return {"trends": trends, "count": len(trends)}

# API Routes for ASES
@api_router.post("/ases", response_model=ASESResponse)
async def create_ases_response(ases_data: ASESCreate):
    # Verify patient exists and has Rotator Cuff injury
    patient = await db.patients.find_one({"id": ases_data.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient_body_part = get_body_part(DiagnosisType(patient.get("diagnosis_type")))
    if patient_body_part != BodyPart.SHOULDER:
        raise HTTPException(status_code=400, detail="ASES questionnaire is only valid for shoulder injuries")
    
    # Create ASES response
    ases_dict = ases_data.model_dump()
    ases_obj = ASESResponse(**ases_dict)
    ases_response_data = ases_obj.model_dump()
    
    # Check if response for this date already exists
    existing = await db.ases_responses.find_one({"patient_id": ases_data.patient_id, "date": ases_data.date})
    if existing:
        # Update existing response
        await db.ases_responses.update_one(
            {"id": existing["id"]},
            {"$set": {**ases_response_data, "updated_at": datetime.utcnow()}}
        )
        ases_response_data["id"] = existing["id"]
        ases_obj = ASESResponse(**ases_response_data)
    else:
        # Insert new response
        await db.ases_responses.insert_one(ases_response_data)
    
    # Calculate scores
    scores_data = calculate_ases_scores(ases_dict)
    scores_obj = ASESScores(
        patient_id=ases_data.patient_id,
        date=ases_data.date,
        ases_response_id=ases_obj.id,
        **scores_data
    )
    scores_data_full = scores_obj.model_dump()
    
    # Check if scores for this date already exist
    existing_scores = await db.ases_scores.find_one({"patient_id": ases_data.patient_id, "date": ases_data.date})
    if existing_scores:
        # Update existing scores
        await db.ases_scores.update_one(
            {"id": existing_scores["id"]},
            {"$set": {**scores_data_full, "updated_at": datetime.utcnow()}}
        )
    else:
        # Insert new scores
        await db.ases_scores.insert_one(scores_data_full)
    
    # Generate updated insights
    await generate_insights(ases_data.patient_id)
    
    return ases_obj

@api_router.get("/ases/{patient_id}", response_model=List[ASESScores])
async def get_patient_ases_scores(patient_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    query = {"patient_id": patient_id}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}
    
    scores = await db.ases_scores.find(query).sort("date", -1).to_list(1000)
    return [ASESScores(**score) for score in scores]

@api_router.get("/ases/{patient_id}/latest", response_model=Optional[ASESScores])
async def get_latest_ases_scores(patient_id: str):
    latest_score = await db.ases_scores.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    if latest_score:
        return ASESScores(**latest_score[0])
    return None

@api_router.get("/ases/{patient_id}/trends")
async def get_ases_trends(patient_id: str, days: int = 90):
    start_date = datetime.utcnow() - timedelta(days=days)
    scores = await db.ases_scores.find({
        "patient_id": patient_id,
        "date": {"$gte": start_date}
    }).sort("date", 1).to_list(1000)
    
    trends = []
    for score in scores:
        trends.append({
            "date": score["date"],
            "pain_component": score["pain_component"],
            "function_component": score["function_component"],
            "total_score": score["total_score"]
        })
    
    return {"trends": trends, "count": len(trends)}

# Enhanced Analytics API Endpoints
@api_router.get("/analytics/{patient_id}/recovery-velocity")
async def get_recovery_velocity_analysis(patient_id: str):
    """Get comprehensive recovery velocity analysis"""
    try:
        from services.wearable_analytics import WearableAnalyticsEngine
        analytics_engine = WearableAnalyticsEngine(db)
        return await analytics_engine.analyze_recovery_velocity(patient_id)
    except ImportError:
        raise HTTPException(status_code=500, detail="Analytics engine not available")

@api_router.get("/analytics/{patient_id}/clinical-risk")
async def get_clinical_risk_assessment(patient_id: str):
    """Get comprehensive clinical risk assessment"""
    try:
        from services.wearable_analytics import WearableAnalyticsEngine
        analytics_engine = WearableAnalyticsEngine(db)
        return await analytics_engine.assess_clinical_risk(patient_id)
    except ImportError:
        raise HTTPException(status_code=500, detail="Analytics engine not available")

@api_router.get("/analytics/{patient_id}/correlations")
async def get_recovery_correlations(patient_id: str):
    """Get comprehensive correlation analysis"""
    try:
        from services.recovery_correlation_engine import RecoveryCorrelationEngine
        correlation_engine = RecoveryCorrelationEngine(db)
        return await correlation_engine.analyze_comprehensive_correlations(patient_id)
    except ImportError:
        raise HTTPException(status_code=500, detail="Correlation engine not available")

@api_router.get("/analytics/{patient_id}/predictions")
async def get_recovery_predictions(patient_id: str):
    """Get recovery timeline and outcome predictions"""
    try:
        from services.predictive_modeling import EnhancedPredictiveModeling
        predictive_engine = EnhancedPredictiveModeling(db)
        
        # Get multiple prediction types
        timeline = await predictive_engine.predict_recovery_timeline(patient_id)
        risk = await predictive_engine.predict_complication_risk(patient_id)
        trajectory = await predictive_engine.predict_pro_score_trajectory(patient_id)
        
        return {
            "timeline_prediction": timeline,
            "complication_risk": risk,
            "pro_trajectory": trajectory
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Predictive modeling not available")

@api_router.get("/analytics/{patient_id}/clinical-alerts")
async def get_clinical_alerts(patient_id: str):
    """Get real-time clinical alerts"""
    try:
        from services.clinical_alerts import ClinicalAlertsEngine
        alerts_engine = ClinicalAlertsEngine(db)
        
        alerts = await alerts_engine.generate_real_time_alerts(patient_id)
        recommendations = await alerts_engine.generate_clinical_recommendations(patient_id)
        intervention_assessment = await alerts_engine.assess_intervention_triggers(patient_id)
        
        return {
            "alerts": [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity.value,
                    "category": alert.alert_type.value,
                    "title": alert.title,
                    "description": alert.description,
                    "recommendations": alert.recommendations,
                    "triggered_at": alert.triggered_at,
                    "requires_immediate_attention": alert.requires_immediate_attention
                }
                for alert in alerts
            ],
            "recommendations": [
                {
                    "id": rec.recommendation_id,
                    "category": rec.category.value,
                    "priority": rec.priority.value,
                    "title": rec.title,
                    "description": rec.description,
                    "action_items": rec.action_items,
                    "timeline": rec.timeline
                }
                for rec in recommendations
            ],
            "intervention_assessment": intervention_assessment
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Clinical alerts engine not available")

@api_router.get("/analytics/{patient_id}/plateau-risk")
async def get_plateau_risk_analysis(patient_id: str):
    """Get plateau risk analysis and prevention strategies"""
    try:
        from services.predictive_modeling import EnhancedPredictiveModeling
        from services.wearable_analytics import WearableAnalyticsEngine
        
        predictive_engine = EnhancedPredictiveModeling(db)
        analytics_engine = WearableAnalyticsEngine(db)
        
        plateau_risk = await predictive_engine.predict_plateau_risk(patient_id)
        plateau_patterns = await analytics_engine.detect_plateau_patterns(patient_id)
        
        return {
            "plateau_risk": plateau_risk,
            "plateau_patterns": plateau_patterns
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Analytics engines not available")

@api_router.get("/analytics/{patient_id}/personalized-insights")
async def get_personalized_insights(patient_id: str):
    """Get personalized recovery insights"""
    try:
        from services.wearable_analytics import WearableAnalyticsEngine
        analytics_engine = WearableAnalyticsEngine(db)
        return await analytics_engine.generate_personalized_insights(patient_id)
    except ImportError:
        raise HTTPException(status_code=500, detail="Analytics engine not available")

@api_router.get("/analytics/{patient_id}/provider-dashboard")
async def get_provider_dashboard_analytics(patient_id: str):
    """Get comprehensive analytics for provider dashboard"""
    try:
        from services.wearable_analytics import WearableAnalyticsEngine
        from services.clinical_alerts import ClinicalAlertsEngine
        
        analytics_engine = WearableAnalyticsEngine(db)
        alerts_engine = ClinicalAlertsEngine(db)
        
        dashboard_metrics = await analytics_engine.analyze_provider_dashboard_metrics(patient_id)
        provider_notifications = await alerts_engine.generate_provider_notifications(patient_id)
        
        return {
            "dashboard_metrics": dashboard_metrics,
            "provider_notifications": provider_notifications
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Analytics engines not available")

@api_router.get("/analytics/population/{diagnosis_type}")
async def get_population_insights(diagnosis_type: str, n_patients: int = 100):
    """Get population-level insights for research and benchmarking"""
    try:
        from services.predictive_modeling import EnhancedPredictiveModeling
        predictive_engine = EnhancedPredictiveModeling(db)
        return await predictive_engine.generate_population_insights(diagnosis_type, n_patients)
    except ImportError:
        raise HTTPException(status_code=500, detail="Predictive modeling not available")

# Quick status endpoint
@api_router.get("/")
async def root():
    return {"message": "Orthopedic Recovery Tracker API is running"}

# Include the routers in the main app
app.include_router(api_router)
app.include_router(wearable_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_db():
    """Initialize database schemas and indexes on startup"""
    try:
        await initialize_wearable_schemas(db)
        # Set database reference for wearable router
        set_database(db)
        logger.info("Database schemas initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database schemas: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
