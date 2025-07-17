"""
MongoDB schemas and database operations for comprehensive wearable data.
Includes indexes, aggregation pipelines, and validation schemas.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


class WearableDataSchemas:
    """MongoDB schemas and operations for wearable data collections"""
    
    @staticmethod
    async def create_indexes(db: AsyncIOMotorDatabase):
        """Create optimized indexes for wearable data collections"""
        
        # Main wearable data collection indexes
        wearable_indexes = [
            # Primary queries
            [("patient_id", 1), ("date", -1)],  # Most common query pattern
            [("patient_id", 1), ("created_at", -1)],  # Latest data queries
            
            # Date range queries
            [("date", 1)],
            [("date", -1)],
            
            # Data quality and source filtering
            [("data_metadata.source", 1)],
            [("data_metadata.data_quality", 1)],
            
            # Activity metrics queries
            [("patient_id", 1), ("activity_metrics.steps", -1)],
            [("patient_id", 1), ("activity_metrics.active_minutes", -1)],
            
            # Heart rate queries
            [("patient_id", 1), ("heart_rate_metrics.resting_hr", 1)],
            [("patient_id", 1), ("heart_rate_metrics.hr_variability_ms", -1)],
            
            # Sleep metrics queries
            [("patient_id", 1), ("sleep_metrics.sleep_efficiency", -1)],
            [("patient_id", 1), ("sleep_metrics.total_sleep_minutes", -1)],
            
            # Movement metrics queries
            [("patient_id", 1), ("movement_metrics.walking_speed_ms", -1)],
            [("patient_id", 1), ("movement_metrics.mobility_score", -1)],
        ]
        
        # Create indexes for main collection
        collection = db.comprehensive_wearable_data
        for index in wearable_indexes:
            try:
                await collection.create_index(index)
                logger.info(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index creation failed for {index}: {e}")
        
        # Trends collection indexes
        trends_indexes = [
            [("patient_id", 1), ("metric_name", 1), ("start_date", -1)],
            [("patient_id", 1), ("end_date", -1)],
            [("metric_name", 1)],
        ]
        
        trends_collection = db.wearable_data_trends
        for index in trends_indexes:
            try:
                await trends_collection.create_index(index)
                logger.info(f"Created trends index: {index}")
            except Exception as e:
                logger.warning(f"Trends index creation failed for {index}: {e}")
        
        # Recovery indicators indexes
        recovery_indexes = [
            [("patient_id", 1), ("analysis_date", -1)],
            [("activity_drop_alert", 1)],
            [("sleep_disruption_alert", 1)],
            [("sedentary_time_alert", 1)],
        ]
        
        recovery_collection = db.recovery_indicators
        for index in recovery_indexes:
            try:
                await recovery_collection.create_index(index)
                logger.info(f"Created recovery indicators index: {index}")
            except Exception as e:
                logger.warning(f"Recovery indicators index creation failed for {index}: {e}")

    @staticmethod
    def get_validation_schema() -> Dict[str, Any]:
        """MongoDB validation schema for wearable data"""
        return {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["patient_id", "date", "created_at"],
                "properties": {
                    "patient_id": {
                        "bsonType": "string",
                        "description": "Patient ID must be a string"
                    },
                    "date": {
                        "bsonType": "date",
                        "description": "Date must be a valid date"
                    },
                    "activity_metrics": {
                        "bsonType": ["object", "null"],
                        "properties": {
                            "steps": {
                                "bsonType": ["int", "null"],
                                "minimum": 0,
                                "maximum": 100000
                            },
                            "distance_meters": {
                                "bsonType": ["double", "null"],
                                "minimum": 0,
                                "maximum": 100000
                            },
                            "calories_active": {
                                "bsonType": ["double", "null"],
                                "minimum": 0,
                                "maximum": 10000
                            }
                        }
                    },
                    "heart_rate_metrics": {
                        "bsonType": ["object", "null"],
                        "properties": {
                            "resting_hr": {
                                "bsonType": ["int", "null"],
                                "minimum": 30,
                                "maximum": 120
                            },
                            "max_hr": {
                                "bsonType": ["int", "null"],
                                "minimum": 30,
                                "maximum": 220
                            }
                        }
                    },
                    "sleep_metrics": {
                        "bsonType": ["object", "null"],
                        "properties": {
                            "total_sleep_minutes": {
                                "bsonType": ["int", "null"],
                                "minimum": 0,
                                "maximum": 1440
                            },
                            "sleep_efficiency": {
                                "bsonType": ["double", "null"],
                                "minimum": 0,
                                "maximum": 100
                            }
                        }
                    }
                }
            }
        }


class WearableDataAggregations:
    """Aggregation pipelines for wearable data analytics"""
    
    @staticmethod
    def get_daily_summary_pipeline(patient_id: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get daily summary statistics for a patient"""
        return [
            {
                "$match": {
                    "patient_id": patient_id,
                    "date": {
                        "$gte": datetime.combine(start_date, datetime.min.time()),
                        "$lte": datetime.combine(end_date, datetime.min.time())
                    }
                }
            },
            {
                "$project": {
                    "date": 1,
                    "steps": "$activity_metrics.steps",
                    "active_minutes": "$activity_metrics.active_minutes",
                    "resting_hr": "$heart_rate_metrics.resting_hr",
                    "sleep_efficiency": "$sleep_metrics.sleep_efficiency",
                    "walking_speed": "$movement_metrics.walking_speed_ms",
                    "mobility_score": "$movement_metrics.mobility_score"
                }
            },
            {
                "$sort": {"date": 1}
            }
        ]
    
    @staticmethod
    def get_weekly_trends_pipeline(patient_id: str, weeks_back: int = 12) -> List[Dict[str, Any]]:
        """Get weekly trends for key metrics"""
        start_date = datetime.utcnow() - timedelta(weeks=weeks_back)
        
        return [
            {
                "$match": {
                    "patient_id": patient_id,
                    "date": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$date"},
                        "week": {"$week": "$date"}
                    },
                    "avg_steps": {"$avg": "$activity_metrics.steps"},
                    "avg_active_minutes": {"$avg": "$activity_metrics.active_minutes"},
                    "avg_resting_hr": {"$avg": "$heart_rate_metrics.resting_hr"},
                    "avg_sleep_efficiency": {"$avg": "$sleep_metrics.sleep_efficiency"},
                    "avg_walking_speed": {"$avg": "$movement_metrics.walking_speed_ms"},
                    "avg_mobility_score": {"$avg": "$movement_metrics.mobility_score"},
                    "week_start": {"$min": "$date"},
                    "data_points": {"$sum": 1}
                }
            },
            {
                "$sort": {"week_start": 1}
            }
        ]
    
    @staticmethod
    def get_activity_patterns_pipeline(patient_id: str) -> List[Dict[str, Any]]:
        """Analyze activity patterns by day of week and hour"""
        return [
            {
                "$match": {"patient_id": patient_id}
            },
            {
                "$unwind": "$exercise_sessions"
            },
            {
                "$group": {
                    "_id": {
                        "day_of_week": {"$dayOfWeek": "$exercise_sessions.start_time"},
                        "hour": {"$hour": "$exercise_sessions.start_time"}
                    },
                    "session_count": {"$sum": 1},
                    "avg_duration": {"$avg": "$exercise_sessions.duration_minutes"},
                    "avg_calories": {"$avg": "$exercise_sessions.calories_burned"}
                }
            },
            {
                "$sort": {"_id.day_of_week": 1, "_id.hour": 1}
            }
        ]
    
    @staticmethod
    def get_recovery_correlation_pipeline(patient_id: str) -> List[Dict[str, Any]]:
        """Analyze correlations between wearable metrics and recovery indicators"""
        return [
            {
                "$match": {"patient_id": patient_id}
            },
            {
                "$lookup": {
                    "from": "koos_responses",  # or ases_responses
                    "localField": "patient_id",
                    "foreignField": "patient_id",
                    "as": "pro_scores"
                }
            },
            {
                "$lookup": {
                    "from": "surveys",
                    "localField": "patient_id", 
                    "foreignField": "patient_id",
                    "as": "surveys"
                }
            },
            {
                "$project": {
                    "date": 1,
                    "steps": "$activity_metrics.steps",
                    "sleep_efficiency": "$sleep_metrics.sleep_efficiency",
                    "resting_hr": "$heart_rate_metrics.resting_hr",
                    "walking_speed": "$movement_metrics.walking_speed_ms",
                    "pro_scores": 1,
                    "surveys": 1
                }
            }
        ]
    
    @staticmethod
    def get_decline_detection_pipeline(patient_id: str, metric: str, days_back: int = 14) -> List[Dict[str, Any]]:
        """Detect declining trends in specific metrics"""
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        metric_path = {
            "steps": "$activity_metrics.steps",
            "walking_speed": "$movement_metrics.walking_speed_ms",
            "sleep_efficiency": "$sleep_metrics.sleep_efficiency",
            "mobility_score": "$movement_metrics.mobility_score"
        }.get(metric, "$activity_metrics.steps")
        
        return [
            {
                "$match": {
                    "patient_id": patient_id,
                    "date": {"$gte": start_date},
                    metric_path.split('.')[1]: {"$exists": True, "$ne": None}
                }
            },
            {
                "$project": {
                    "date": 1,
                    "metric_value": metric_path,
                    "day_number": {
                        "$dateDiff": {
                            "startDate": start_date,
                            "endDate": "$date",
                            "unit": "day"
                        }
                    }
                }
            },
            {
                "$sort": {"date": 1}
            },
            {
                "$group": {
                    "_id": None,
                    "values": {"$push": "$metric_value"},
                    "dates": {"$push": "$date"},
                    "day_numbers": {"$push": "$day_number"},
                    "first_value": {"$first": "$metric_value"},
                    "last_value": {"$last": "$metric_value"},
                    "avg_value": {"$avg": "$metric_value"},
                    "count": {"$sum": 1}
                }
            }
        ]
    
    @staticmethod
    def get_milestone_tracking_pipeline(patient_id: str) -> List[Dict[str, Any]]:
        """Track progress toward recovery milestones using wearable data"""
        return [
            {
                "$match": {"patient_id": patient_id}
            },
            {
                "$project": {
                    "date": 1,
                    "weeks_from_start": {
                        "$dateDiff": {
                            "startDate": "2024-01-01",  # Will be replaced with surgery date
                            "endDate": "$date",
                            "unit": "week"
                        }
                    },
                    "daily_steps": "$activity_metrics.steps",
                    "walking_speed": "$movement_metrics.walking_speed_ms",
                    "stairs_climbed": "$activity_metrics.floors_climbed",
                    "active_minutes": "$activity_metrics.active_minutes"
                }
            },
            {
                "$group": {
                    "_id": "$weeks_from_start",
                    "avg_daily_steps": {"$avg": "$daily_steps"},
                    "max_walking_speed": {"$max": "$walking_speed"},
                    "total_stairs": {"$sum": "$stairs_climbed"},
                    "avg_active_minutes": {"$avg": "$active_minutes"},
                    "data_points": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]


class WearableDataQueries:
    """Common query operations for wearable data"""
    
    @staticmethod
    async def get_latest_metrics(db: AsyncIOMotorDatabase, patient_id: str, days: int = 7) -> Optional[Dict]:
        """Get latest metrics for a patient"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "patient_id": patient_id,
                    "date": {"$gte": start_date}
                }
            },
            {
                "$sort": {"date": -1}
            },
            {
                "$limit": 1
            }
        ]
        
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        return result[0] if result else None
    
    @staticmethod
    async def check_data_quality(db: AsyncIOMotorDatabase, patient_id: str) -> Dict[str, Any]:
        """Check data quality metrics for a patient"""
        pipeline = [
            {
                "$match": {"patient_id": patient_id}
            },
            {
                "$group": {
                    "_id": "$data_metadata.data_quality",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$data_metadata.confidence_score"}
                }
            }
        ]
        
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        return {
            "quality_distribution": results,
            "total_records": sum(r["count"] for r in results)
        }
    
    @staticmethod
    async def detect_data_gaps(db: AsyncIOMotorDatabase, patient_id: str, days_back: int = 30) -> List[date]:
        """Detect missing data days for a patient"""
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Get all dates with data
        pipeline = [
            {
                "$match": {
                    "patient_id": patient_id,
                    "date": {"$gte": start_date}
                }
            },
            {
                "$group": {
                    "_id": "$date"
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        cursor = db.comprehensive_wearable_data.aggregate(pipeline)
        data_dates = [result["_id"].date() for result in await cursor.to_list(length=None)]
        
        # Generate expected date range
        expected_dates = []
        current_date = start_date.date()
        end_date = datetime.utcnow().date()
        
        while current_date <= end_date:
            expected_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Find missing dates
        missing_dates = [d for d in expected_dates if d not in data_dates]
        return missing_dates


# Database initialization function
async def initialize_wearable_schemas(db: AsyncIOMotorDatabase):
    """Initialize all wearable data schemas and indexes"""
    try:
        # Create indexes
        await WearableDataSchemas.create_indexes(db)
        
        # Set up validation schema for main collection
        validation_schema = WearableDataSchemas.get_validation_schema()
        
        try:
            await db.command({
                "collMod": "comprehensive_wearable_data",
                "validator": validation_schema,
                "validationLevel": "moderate"
            })
            logger.info("Validation schema applied to comprehensive_wearable_data collection")
        except Exception as e:
            logger.warning(f"Failed to apply validation schema: {e}")
        
        logger.info("Wearable data schemas initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize wearable schemas: {e}")
        raise