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
class InjuryType(str, Enum):
    ACL = "ACL"
    ROTATOR_CUFF = "Rotator Cuff"

class RecoveryStatus(str, Enum):
    ON_TRACK = "On Track"
    AT_RISK = "At Risk"
    NEEDS_ATTENTION = "Needs Attention"

# Define Models
class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    injury_type: InjuryType
    date_of_injury: datetime
    date_of_surgery: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PatientCreate(BaseModel):
    name: str
    email: str
    injury_type: InjuryType
    date_of_injury: datetime
    date_of_surgery: Optional[datetime] = None

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
    date: date
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
    date: date
    pain_score: int = Field(ge=0, le=10)
    mobility_score: int = Field(ge=0, le=10)
    activities_of_daily_living: Dict[str, int] = {}
    range_of_motion: Dict[str, float] = {}
    strength: Dict[str, int] = {}
    notes: Optional[str] = None

class AIInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    date: date
    recovery_status: RecoveryStatus
    recommendations: List[str] = []
    risk_factors: List[str] = []
    progress_percentage: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

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

# Helper function to generate AI insights
async def generate_insights(patient_id: str) -> Optional[AIInsight]:
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        return None
    
    # Get latest survey
    latest_survey = await db.surveys.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    if not latest_survey:
        return None
    
    latest_survey = latest_survey[0]
    
    # Get latest wearable data
    latest_wearable = await db.wearable_data.find({"patient_id": patient_id}).sort("date", -1).limit(1).to_list(1)
    
    # Simple AI logic for insights
    recovery_status = RecoveryStatus.ON_TRACK
    recommendations = []
    risk_factors = []
    progress_percentage = 0.0
    
    pain_score = latest_survey.get("pain_score", 0)
    mobility_score = latest_survey.get("mobility_score", 0)
    
    # Calculate recovery status based on pain and mobility
    if pain_score > 7:
        recovery_status = RecoveryStatus.NEEDS_ATTENTION
        risk_factors.append("High pain levels")
        recommendations.append("Consult with your doctor about pain management options")
    elif pain_score > 5:
        recovery_status = RecoveryStatus.AT_RISK
        risk_factors.append("Elevated pain levels")
        recommendations.append("Consider reducing activity and applying ice")
    
    if mobility_score < 3:
        recovery_status = RecoveryStatus.NEEDS_ATTENTION
        risk_factors.append("Very limited mobility")
        recommendations.append("Focus on prescribed mobility exercises")
    elif mobility_score < 5:
        if recovery_status != RecoveryStatus.NEEDS_ATTENTION:
            recovery_status = RecoveryStatus.AT_RISK
        risk_factors.append("Restricted mobility")
        recommendations.append("Gentle stretching may help improve mobility")
    
    # Add wearable data checks if available
    if latest_wearable:
        wearable = latest_wearable[0]
        
        if wearable.get("steps", 0) < 1000:
            if recovery_status != RecoveryStatus.NEEDS_ATTENTION:
                recovery_status = RecoveryStatus.AT_RISK
            risk_factors.append("Very low activity level")
            recommendations.append("Try to increase daily activity gradually as tolerated")
        
        if wearable.get("sleep_hours", 0) < 6:
            risk_factors.append("Insufficient sleep")
            recommendations.append("Adequate sleep is crucial for recovery")
    
    # Calculate progress percentage based on pain and mobility improvement
    # This is a simplified calculation and would be more complex in a real application
    progress_percentage = (10 - pain_score) * 0.4 + mobility_score * 0.6  # 0-10 scale converted to 0-100%
    progress_percentage = min(max(progress_percentage * 10, 0), 100)  # Ensure it's between 0-100
    
    # Create insight object
    insight = AIInsight(
        patient_id=patient_id,
        date=latest_survey.get("date"),
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
        "date": latest_survey.get("date")
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
    # Sample patients
    patients = [
        PatientCreate(
            name="John Smith",
            email="john.smith@example.com",
            injury_type=InjuryType.ACL,
            date_of_injury=datetime.strptime("2025-01-15", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-01", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Sarah Johnson",
            email="sarah.johnson@example.com",
            injury_type=InjuryType.ROTATOR_CUFF,
            date_of_injury=datetime.strptime("2025-02-05", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-02-20", "%Y-%m-%d")
        ),
        PatientCreate(
            name="Michael Brown",
            email="michael.brown@example.com",
            injury_type=InjuryType.ACL,
            date_of_injury=datetime.strptime("2025-01-10", "%Y-%m-%d"),
            date_of_surgery=datetime.strptime("2025-01-25", "%Y-%m-%d")
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
            
            # Vary by patient condition
            if patient.injury_type == InjuryType.ACL:
                # ACL patients might have more restricted mobility initially
                step_multiplier = 0.7
                heart_rate_base = 75
            else:
                # Rotator cuff patients might have less impact on walking
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
            
            # Adjust based on injury type
            if patient.injury_type == InjuryType.ACL:
                # ACL specific values
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
                # Rotator cuff specific values
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

# Quick status endpoint
@api_router.get("/")
async def root():
    return {"message": "Orthopedic Recovery Tracker API is running"}

# Include the router in the main app
app.include_router(api_router)

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

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
