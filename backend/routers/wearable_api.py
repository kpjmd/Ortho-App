"""
Comprehensive FastAPI router for wearable data management in RcvryAI.
Provides advanced CRUD operations, bulk import, analytics, and real-time sync capabilities.
"""

from fastapi import APIRouter, HTTPException, Query, File, UploadFile, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date, timedelta
from pydantic import BaseModel, Field, ValidationError
import json
import csv
import io
import uuid
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

# Set up logging
logger = logging.getLogger(__name__)

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
    PhysiologicalMetrics,
    DataSource,
    DataQuality,
    DataMetadata
)
from schemas.wearable_schemas import (
    WearableDataAggregations,
    WearableDataQueries
)

# Create the router
wearable_router = APIRouter(prefix="/patients", tags=["wearable-data"])

# Database reference (will be injected)
db: AsyncIOMotorDatabase = None

def set_database(database: AsyncIOMotorDatabase):
    """Set the database instance for the router"""
    global db
    db = database

# Error handling utilities
class WearableAPIError(Exception):
    """Custom exception for wearable API errors"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

async def validate_patient_exists(patient_id: str) -> bool:
    """Validate that a patient exists in the database"""
    if not db:
        raise WearableAPIError("Database not initialized", 500)
    
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise WearableAPIError(f"Patient {patient_id} not found", 404)
    return True

def handle_database_error(func):
    """Decorator to handle common database errors"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except WearableAPIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    return wrapper

# Response models for API endpoints
class WearableDataResponse(BaseModel):
    """Standard response wrapper for wearable data operations"""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    errors: List[str] = []

class BulkImportResponse(BaseModel):
    """Response model for bulk import operations"""
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    errors: List[str] = []
    total_processed: int = 0

class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    patient_id: str
    total_records: int
    quality_distribution: Dict[str, int]
    missing_dates: List[date]
    completeness_score: float
    data_sources: Dict[str, int]
    recommendations: List[str]

class WearableSummary(BaseModel):
    """Dashboard summary of wearable data"""
    patient_id: str
    date_range: Dict[str, date]
    total_days: int
    avg_daily_steps: Optional[float] = None
    avg_sleep_hours: Optional[float] = None
    avg_resting_hr: Optional[float] = None
    activity_consistency: Optional[float] = None
    recent_trends: Dict[str, str] = {}
    alerts: List[str] = []

class ExportRequest(BaseModel):
    """Request model for data export"""
    format: str = Field(default="json", regex="^(json|csv|fhir)$")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    include_raw_data: bool = True
    include_aggregated: bool = True
    anonymize: bool = False

# ==================== CORE CRUD OPERATIONS ====================

@wearable_router.post("/{patient_id}/wearable/data", response_model=ComprehensiveWearableData)
@handle_database_error
async def create_wearable_data(
    patient_id: str,
    data: ComprehensiveWearableDataCreate
):
    """Create a comprehensive wearable data entry with rich health metrics"""
    # Verify patient exists
    await validate_patient_exists(patient_id)
    
    # Set patient_id from URL parameter
    data.patient_id = patient_id
    
    # Create wearable data object
    wearable_obj = ComprehensiveWearableData(**data.model_dump())
    wearable_data = wearable_obj.model_dump()
    
    # Check if entry for this date already exists
    existing = await db.comprehensive_wearable_data.find_one({
        "patient_id": patient_id,
        "date": data.date
    })
    
    if existing:
        # Update existing entry
        await db.comprehensive_wearable_data.update_one(
            {"id": existing["id"]},
            {"$set": {**wearable_data, "updated_at": datetime.utcnow()}}
        )
        wearable_data["id"] = existing["id"]
        return ComprehensiveWearableData(**wearable_data)
    
    # Insert new entry
    await db.comprehensive_wearable_data.insert_one(wearable_data)
    
    return wearable_obj

@wearable_router.get("/{patient_id}/wearable/data", response_model=List[ComprehensiveWearableData])
@handle_database_error
async def get_wearable_data(
    patient_id: str,
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    data_source: Optional[DataSource] = Query(None, description="Filter by data source"),
    quality_threshold: Optional[DataQuality] = Query(None, description="Minimum data quality"),
    limit: int = Query(100, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip")
):
    """Get wearable data with advanced filtering options"""
    try:
        # Build query
        query = {"patient_id": patient_id}
        
        # Date filtering
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["date"] = date_filter
        
        # Data source filtering
        if data_source:
            query["data_metadata.source"] = data_source
        
        # Quality filtering
        if quality_threshold:
            quality_order = ["POOR", "LOW", "MEDIUM", "HIGH"]
            min_index = quality_order.index(quality_threshold.value)
            acceptable_qualities = quality_order[min_index:]
            query["data_metadata.data_quality"] = {"$in": acceptable_qualities}
        
        # Execute query
        cursor = db.comprehensive_wearable_data.find(query)\
            .sort("date", -1)\
            .skip(offset)\
            .limit(limit)
        
        data = await cursor.to_list(length=limit)
        return [ComprehensiveWearableData(**item) for item in data]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve wearable data: {str(e)}")

@wearable_router.get("/{patient_id}/wearable/data/{entry_id}", response_model=ComprehensiveWearableData)
async def get_wearable_entry(patient_id: str, entry_id: str):
    """Get a specific wearable data entry"""
    try:
        entry = await db.comprehensive_wearable_data.find_one({
            "id": entry_id,
            "patient_id": patient_id
        })
        
        if not entry:
            raise HTTPException(status_code=404, detail="Wearable data entry not found")
        
        return ComprehensiveWearableData(**entry)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve entry: {str(e)}")

@wearable_router.put("/{patient_id}/wearable/data/{entry_id}", response_model=ComprehensiveWearableData)
async def update_wearable_entry(
    patient_id: str,
    entry_id: str,
    data: ComprehensiveWearableDataCreate
):
    """Update a specific wearable data entry"""
    try:
        # Verify entry exists
        existing = await db.comprehensive_wearable_data.find_one({
            "id": entry_id,
            "patient_id": patient_id
        })
        
        if not existing:
            raise HTTPException(status_code=404, detail="Wearable data entry not found")
        
        # Set patient_id from URL parameter
        data.patient_id = patient_id
        
        # Create updated object
        wearable_obj = ComprehensiveWearableData(**data.model_dump())
        wearable_obj.id = entry_id
        wearable_obj.created_at = existing["created_at"]
        wearable_obj.updated_at = datetime.utcnow()
        
        # Update in database
        wearable_data = wearable_obj.model_dump()
        await db.comprehensive_wearable_data.update_one(
            {"id": entry_id},
            {"$set": wearable_data}
        )
        
        return wearable_obj
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")

@wearable_router.delete("/{patient_id}/wearable/data/{entry_id}")
async def delete_wearable_entry(patient_id: str, entry_id: str):
    """Delete a specific wearable data entry"""
    try:
        result = await db.comprehensive_wearable_data.delete_one({
            "id": entry_id,
            "patient_id": patient_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Wearable data entry not found")
        
        return {"message": "Entry deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete entry: {str(e)}")

# ==================== BULK OPERATIONS ====================

@wearable_router.post("/{patient_id}/wearable/bulk-import", response_model=BulkImportResponse)
@handle_database_error
async def bulk_import_wearable_data(
    patient_id: str,
    import_data: BulkWearableDataImport
):
    """Bulk import wearable data from external sources"""
    try:
        # Verify patient exists
        patient = await db.patients.find_one({"id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        imported_count = 0
        updated_count = 0
        skipped_count = 0
        errors = []
        
        for data_record in import_data.data_records:
            try:
                # Set patient_id and metadata
                data_record.patient_id = patient_id
                if data_record.data_metadata is None:
                    data_record.data_metadata = DataMetadata(
                        source=import_data.import_source,
                        data_quality=DataQuality.MEDIUM
                    )
                
                wearable_obj = ComprehensiveWearableData(**data_record.model_dump())
                wearable_data = wearable_obj.model_dump()
                
                # Check if entry exists
                existing = await db.comprehensive_wearable_data.find_one({
                    "patient_id": patient_id,
                    "date": data_record.date
                })
                
                if existing:
                    # Check if data is different
                    if _data_needs_update(existing, wearable_data):
                        await db.comprehensive_wearable_data.update_one(
                            {"id": existing["id"]},
                            {"$set": {**wearable_data, "updated_at": datetime.utcnow()}}
                        )
                        updated_count += 1
                    else:
                        skipped_count += 1
                else:
                    await db.comprehensive_wearable_data.insert_one(wearable_data)
                    imported_count += 1
                    
            except Exception as e:
                errors.append(f"Date {data_record.date}: {str(e)}")
        
        return BulkImportResponse(
            imported=imported_count,
            updated=updated_count,
            skipped=skipped_count,
            errors=errors,
            total_processed=len(import_data.data_records)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk import failed: {str(e)}")

@wearable_router.post("/{patient_id}/wearable/bulk-import/file")
async def bulk_import_from_file(
    patient_id: str,
    file: UploadFile = File(...),
    data_source: DataSource = DataSource.UNKNOWN
):
    """Import wearable data from uploaded file (CSV or JSON)"""
    try:
        # Verify patient exists
        patient = await db.patients.find_one({"id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Read file content
        content = await file.read()
        
        # Parse based on file type
        if file.filename.endswith('.csv'):
            data_records = await _parse_csv_data(content, patient_id)
        elif file.filename.endswith('.json'):
            data_records = await _parse_json_data(content, patient_id)
        elif file.filename.endswith('.xml'):
            data_records = await _parse_healthkit_data(content, patient_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, JSON, or XML.")
        
        # Create bulk import request
        import_request = BulkWearableDataImport(
            patient_id=patient_id,
            data_records=data_records,
            import_source=data_source,
            import_notes=f"Imported from file: {file.filename}"
        )
        
        # Process bulk import
        return await bulk_import_wearable_data(patient_id, import_request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File import failed: {str(e)}")

@wearable_router.post("/{patient_id}/wearable/sync", response_model=WearableDataResponse)
async def sync_wearable_data(
    patient_id: str,
    sync_data: List[ComprehensiveWearableDataCreate],
    device_id: Optional[str] = None,
    last_sync: Optional[datetime] = None
):
    """Real-time sync endpoint optimized for mobile apps"""
    try:
        # Verify patient exists
        patient = await db.patients.find_one({"id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        synced_count = 0
        conflicts = []
        
        for data_entry in sync_data:
            data_entry.patient_id = patient_id
            
            # Check for conflicts if last_sync is provided
            if last_sync:
                existing = await db.comprehensive_wearable_data.find_one({
                    "patient_id": patient_id,
                    "date": data_entry.date,
                    "updated_at": {"$gt": last_sync}
                })
                
                if existing:
                    conflicts.append({
                        "date": data_entry.date,
                        "conflict_type": "server_newer",
                        "server_updated": existing["updated_at"]
                    })
                    continue
            
            # Upsert data
            wearable_obj = ComprehensiveWearableData(**data_entry.model_dump())
            await db.comprehensive_wearable_data.update_one(
                {"patient_id": patient_id, "date": data_entry.date},
                {"$set": wearable_obj.model_dump()},
                upsert=True
            )
            synced_count += 1
        
        return WearableDataResponse(
            message=f"Synced {synced_count} entries",
            data={
                "synced": synced_count,
                "conflicts": conflicts,
                "sync_timestamp": datetime.utcnow()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

# ==================== ANALYTICS & INSIGHTS ====================

@wearable_router.get("/{patient_id}/wearable/summary", response_model=WearableSummary)
async def get_wearable_summary(
    patient_id: str,
    days_back: int = Query(30, le=365, description="Number of days to include in summary")
):
    """Get comprehensive dashboard summary of wearable data"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days_back)
        end_date = datetime.utcnow()
        
        # Get summary statistics
        pipeline = WearableDataAggregations.get_daily_summary_pipeline(
            patient_id, start_date.date(), end_date.date()
        )
        
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        daily_data = await cursor.to_list(length=None)
        
        if not daily_data:
            raise HTTPException(status_code=404, detail="No wearable data found for patient")
        
        # Calculate summary metrics
        total_days = len(daily_data)
        avg_steps = sum(d.get("steps", 0) or 0 for d in daily_data) / total_days if total_days > 0 else 0
        
        sleep_data = [d.get("sleep_efficiency") for d in daily_data if d.get("sleep_efficiency")]
        avg_sleep_efficiency = sum(sleep_data) / len(sleep_data) if sleep_data else None
        
        hr_data = [d.get("resting_hr") for d in daily_data if d.get("resting_hr")]
        avg_resting_hr = sum(hr_data) / len(hr_data) if hr_data else None
        
        # Calculate activity consistency (coefficient of variation)
        step_data = [d.get("steps", 0) or 0 for d in daily_data]
        if len(step_data) > 1 and avg_steps > 0:
            step_variance = sum((s - avg_steps) ** 2 for s in step_data) / len(step_data)
            step_std = step_variance ** 0.5
            activity_consistency = 1 - (step_std / avg_steps)
        else:
            activity_consistency = None
        
        # Detect recent trends
        recent_trends = await _analyze_recent_trends(patient_id, daily_data[-7:] if len(daily_data) >= 7 else daily_data)
        
        # Generate alerts
        alerts = await _generate_summary_alerts(patient_id, daily_data)
        
        return WearableSummary(
            patient_id=patient_id,
            date_range={
                "start": start_date.date(),
                "end": end_date.date()
            },
            total_days=total_days,
            avg_daily_steps=avg_steps,
            avg_sleep_hours=avg_sleep_efficiency,
            avg_resting_hr=avg_resting_hr,
            activity_consistency=activity_consistency,
            recent_trends=recent_trends,
            alerts=alerts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@wearable_router.get("/{patient_id}/wearable/trends")
async def get_wearable_trends(
    patient_id: str,
    weeks_back: int = Query(12, le=52, description="Number of weeks for trend analysis"),
    metrics: List[str] = Query(["steps", "sleep_efficiency", "walking_speed"], description="Metrics to analyze")
):
    """Get advanced trend analysis for wearable metrics"""
    try:
        pipeline = WearableDataAggregations.get_weekly_trends_pipeline(patient_id, weeks_back)
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        weekly_data = await cursor.to_list(length=None)
        
        trends = {}
        for metric in metrics:
            metric_key = f"avg_{metric}"
            if metric_key in (weekly_data[0] if weekly_data else {}):
                trend_data = [week.get(metric_key) for week in weekly_data if week.get(metric_key) is not None]
                if len(trend_data) >= 3:
                    trends[metric] = _calculate_trend_direction(trend_data)
        
        return {
            "patient_id": patient_id,
            "weeks_analyzed": len(weekly_data),
            "weekly_data": weekly_data,
            "trend_analysis": trends,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

@wearable_router.get("/{patient_id}/wearable/recovery-insights", response_model=RecoveryIndicators)
async def get_recovery_insights(patient_id: str):
    """Get AI-enhanced recovery indicators based on wearable data"""
    try:
        # Use existing recovery analysis function from server.py
        from server import analyze_recovery_indicators
        indicators = await analyze_recovery_indicators(patient_id)
        
        if "error" in indicators:
            raise HTTPException(status_code=404, detail=indicators["error"])
        
        return RecoveryIndicators(**indicators)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recovery insights: {str(e)}")

@wearable_router.get("/{patient_id}/wearable/correlations")
async def get_wearable_correlations(
    patient_id: str,
    days_back: int = Query(90, le=365, description="Days of data to analyze")
):
    """Analyze correlations between wearable metrics and PRO scores"""
    try:
        pipeline = WearableDataAggregations.get_recovery_correlation_pipeline(patient_id)
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        correlation_data = await cursor.to_list(length=None)
        
        # Calculate correlations with PRO scores
        correlations = await _calculate_pro_correlations(patient_id, correlation_data)
        
        return {
            "patient_id": patient_id,
            "analysis_period_days": days_back,
            "correlations": correlations,
            "data_points": len(correlation_data),
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate correlations: {str(e)}")

# ==================== DATA QUALITY & EXPORT ====================

@wearable_router.get("/{patient_id}/wearable/data-quality", response_model=DataQualityReport)
async def get_data_quality_report(patient_id: str):
    """Get comprehensive data quality assessment"""
    try:
        # Get quality metrics
        quality_metrics = await WearableDataQueries.check_data_quality(db, patient_id)
        
        # Detect missing dates
        missing_dates = await WearableDataQueries.detect_data_gaps(db, patient_id, days_back=30)
        
        # Calculate completeness score
        expected_days = 30
        missing_days = len(missing_dates)
        completeness_score = max(0, (expected_days - missing_days) / expected_days * 100)
        
        # Get data source distribution
        source_pipeline = [
            {"$match": {"patient_id": patient_id}},
            {"$group": {"_id": "$data_metadata.source", "count": {"$sum": 1}}}
        ]
        cursor = db.comprehensive_wearable_data.aggregate(source_pipeline)
        source_data = await cursor.to_list(length=None)
        data_sources = {item["_id"]: item["count"] for item in source_data}
        
        # Generate recommendations
        recommendations = _generate_quality_recommendations(
            quality_metrics, missing_dates, completeness_score
        )
        
        return DataQualityReport(
            patient_id=patient_id,
            total_records=quality_metrics.get("total_records", 0),
            quality_distribution={
                item["_id"]: item["count"] 
                for item in quality_metrics.get("quality_distribution", [])
            },
            missing_dates=missing_dates,
            completeness_score=completeness_score,
            data_sources=data_sources,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quality report: {str(e)}")

@wearable_router.get("/{patient_id}/wearable/export")
async def export_wearable_data(
    patient_id: str,
    export_request: ExportRequest = ExportRequest()
):
    """Export wearable data in various formats for research/insurance"""
    try:
        # Build query for data export
        query = {"patient_id": patient_id}
        if export_request.start_date or export_request.end_date:
            date_filter = {}
            if export_request.start_date:
                date_filter["$gte"] = export_request.start_date
            if export_request.end_date:
                date_filter["$lte"] = export_request.end_date
            query["date"] = date_filter
        
        # Get raw data if requested
        export_data = {}
        if export_request.include_raw_data:
            cursor = db.comprehensive_wearable_data.find(query).sort("date", 1)
            raw_data = await cursor.to_list(length=None)
            
            if export_request.anonymize:
                raw_data = _anonymize_data(raw_data)
            
            export_data["raw_data"] = raw_data
        
        # Get aggregated data if requested
        if export_request.include_aggregated:
            # Weekly aggregations
            pipeline = WearableDataAggregations.get_weekly_trends_pipeline(patient_id, 52)
            cursor = db.comprehensive_wearable_data.aggregate(pipeline)
            export_data["weekly_aggregates"] = await cursor.to_list(length=None)
        
        # Format based on requested format
        if export_request.format == "csv":
            return _format_as_csv(export_data)
        elif export_request.format == "fhir":
            return _format_as_fhir(export_data, patient_id)
        else:  # JSON
            return {
                "patient_id": patient_id if not export_request.anonymize else "ANONYMIZED",
                "export_timestamp": datetime.utcnow(),
                "format": export_request.format,
                **export_data
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# ==================== HELPER FUNCTIONS ====================

def _data_needs_update(existing: Dict, new_data: Dict) -> bool:
    """Check if new data differs significantly from existing data"""
    # Compare key metrics to determine if update is needed
    key_fields = ["activity_metrics", "heart_rate_metrics", "sleep_metrics", "movement_metrics"]
    
    for field in key_fields:
        if existing.get(field) != new_data.get(field):
            return True
    
    return False

async def _parse_csv_data(content: bytes, patient_id: str) -> List[ComprehensiveWearableDataCreate]:
    """Parse CSV data and convert to wearable data records"""
    # Implementation for CSV parsing
    # This would include mapping CSV columns to our data model
    data_records = []
    
    try:
        csv_content = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        for row in csv_reader:
            # Map CSV columns to our model
            # This is a simplified example - real implementation would need robust field mapping
            record = ComprehensiveWearableDataCreate(
                patient_id=patient_id,
                date=datetime.strptime(row.get('date', ''), '%Y-%m-%d').date(),
                activity_metrics=ActivityMetrics(
                    steps=int(row.get('steps', 0)) if row.get('steps') else None,
                    distance_meters=float(row.get('distance', 0)) if row.get('distance') else None
                ) if row.get('steps') or row.get('distance') else None
            )
            data_records.append(record)
    
    except Exception as e:
        raise ValueError(f"CSV parsing error: {str(e)}")
    
    return data_records

async def _parse_json_data(content: bytes, patient_id: str) -> List[ComprehensiveWearableDataCreate]:
    """Parse JSON data and convert to wearable data records"""
    try:
        json_data = json.loads(content.decode('utf-8'))
        data_records = []
        
        # Handle different JSON structures
        if isinstance(json_data, list):
            for item in json_data:
                record = ComprehensiveWearableDataCreate(patient_id=patient_id, **item)
                data_records.append(record)
        elif isinstance(json_data, dict) and 'data' in json_data:
            for item in json_data['data']:
                record = ComprehensiveWearableDataCreate(patient_id=patient_id, **item)
                data_records.append(record)
        
        return data_records
    
    except Exception as e:
        raise ValueError(f"JSON parsing error: {str(e)}")

async def _parse_healthkit_data(content: bytes, patient_id: str) -> List[ComprehensiveWearableDataCreate]:
    """Parse HealthKit XML export and convert to wearable data records"""
    # This would require XML parsing and HealthKit data structure knowledge
    # Placeholder implementation
    raise NotImplementedError("HealthKit XML parsing not yet implemented")

async def _analyze_recent_trends(patient_id: str, recent_data: List[Dict]) -> Dict[str, str]:
    """Analyze recent trends in wearable data"""
    trends = {}
    
    if len(recent_data) >= 3:
        # Analyze steps trend
        steps_data = [d.get("steps", 0) or 0 for d in recent_data]
        trends["steps"] = _calculate_trend_direction(steps_data)
        
        # Analyze sleep trend
        sleep_data = [d.get("sleep_efficiency") for d in recent_data if d.get("sleep_efficiency")]
        if len(sleep_data) >= 3:
            trends["sleep"] = _calculate_trend_direction(sleep_data)
    
    return trends

def _calculate_trend_direction(data: List[float]) -> str:
    """Calculate trend direction from data points"""
    if len(data) < 3:
        return "insufficient_data"
    
    # Simple linear trend calculation
    x = list(range(len(data)))
    n = len(data)
    
    sum_x = sum(x)
    sum_y = sum(data)
    sum_xy = sum(xi * yi for xi, yi in zip(x, data))
    sum_x2 = sum(xi * xi for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    if slope > 0.1:
        return "improving"
    elif slope < -0.1:
        return "declining"
    else:
        return "stable"

async def _generate_summary_alerts(patient_id: str, daily_data: List[Dict]) -> List[str]:
    """Generate alerts based on daily data patterns"""
    alerts = []
    
    if len(daily_data) >= 7:
        # Check for low activity
        recent_steps = [d.get("steps", 0) or 0 for d in daily_data[-7:]]
        avg_recent_steps = sum(recent_steps) / len(recent_steps)
        
        if avg_recent_steps < 2000:
            alerts.append("Low activity levels detected in past week")
        
        # Check for sleep issues
        sleep_data = [d.get("sleep_efficiency") for d in daily_data[-7:] if d.get("sleep_efficiency")]
        if sleep_data and sum(sleep_data) / len(sleep_data) < 75:
            alerts.append("Poor sleep efficiency detected")
    
    return alerts

async def _calculate_pro_correlations(patient_id: str, wearable_data: List[Dict]) -> Dict[str, float]:
    """Calculate correlations between wearable metrics and PRO scores"""
    correlations = {}
    
    # This would implement correlation analysis with KOOS/ASES scores
    # Placeholder implementation
    correlations["steps_vs_pain"] = 0.0
    correlations["sleep_vs_function"] = 0.0
    
    return correlations

def _generate_quality_recommendations(quality_metrics: Dict, missing_dates: List[date], completeness_score: float) -> List[str]:
    """Generate data quality improvement recommendations"""
    recommendations = []
    
    if completeness_score < 80:
        recommendations.append("Consider more consistent data collection")
    
    if len(missing_dates) > 7:
        recommendations.append("Large data gaps detected - check device connectivity")
    
    return recommendations

def _anonymize_data(data: List[Dict]) -> List[Dict]:
    """Remove personally identifiable information from data"""
    anonymized = []
    for record in data:
        clean_record = record.copy()
        clean_record.pop("patient_id", None)
        clean_record.pop("id", None)
        anonymized.append(clean_record)
    return anonymized

def _format_as_csv(data: Dict) -> str:
    """Format export data as CSV"""
    # Implementation for CSV formatting
    return "CSV export not yet implemented"

def _format_as_fhir(data: Dict, patient_id: str) -> Dict:
    """Format export data as FHIR resources"""
    # Implementation for FHIR formatting
    return {"message": "FHIR export not yet implemented"}