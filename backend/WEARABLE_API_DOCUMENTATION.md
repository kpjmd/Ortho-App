# Comprehensive Wearable Data API Documentation

## Overview
This document describes the comprehensive wearable data management API endpoints implemented in `routers/wearable_api.py` for the RcvryAI orthopedic recovery tracking application.

## API Endpoints

### Core CRUD Operations

#### Create Wearable Data
- **POST** `/api/patients/{patient_id}/wearable/data`
- Creates a comprehensive wearable data entry with rich health metrics
- **Body**: `ComprehensiveWearableDataCreate` model
- **Response**: `ComprehensiveWearableData`

#### Get Wearable Data
- **GET** `/api/patients/{patient_id}/wearable/data`
- Retrieves wearable data with advanced filtering options
- **Query Parameters**:
  - `start_date`: Start date for filtering
  - `end_date`: End date for filtering
  - `data_source`: Filter by data source (HealthKit, Fitbit, etc.)
  - `quality_threshold`: Minimum data quality filter
  - `limit`: Maximum number of records (default: 100, max: 1000)
  - `offset`: Number of records to skip

#### Get Specific Entry
- **GET** `/api/patients/{patient_id}/wearable/data/{entry_id}`
- Retrieves a specific wearable data entry
- **Response**: `ComprehensiveWearableData`

#### Update Entry
- **PUT** `/api/patients/{patient_id}/wearable/data/{entry_id}`
- Updates a specific wearable data entry
- **Body**: `ComprehensiveWearableDataCreate` model

#### Delete Entry
- **DELETE** `/api/patients/{patient_id}/wearable/data/{entry_id}`
- Removes a wearable data entry

### Bulk Operations

#### Bulk Import
- **POST** `/api/patients/{patient_id}/wearable/bulk-import`
- Bulk imports wearable data from external sources
- **Body**: `BulkWearableDataImport` model
- **Response**: `BulkImportResponse` with counts and errors

#### File Import
- **POST** `/api/patients/{patient_id}/wearable/bulk-import/file`
- Imports wearable data from uploaded files (CSV, JSON, XML)
- **Parameters**:
  - `file`: Upload file
  - `data_source`: Source type (HealthKit, Fitbit, etc.)

#### Real-time Sync
- **POST** `/api/patients/{patient_id}/wearable/sync`
- Optimized sync endpoint for mobile apps
- **Body**: List of `ComprehensiveWearableDataCreate`
- **Query Parameters**:
  - `device_id`: Optional device identifier
  - `last_sync`: Timestamp for conflict detection

### Analytics & Insights

#### Summary Dashboard
- **GET** `/api/patients/{patient_id}/wearable/summary`
- Comprehensive dashboard summary of wearable data
- **Query Parameters**:
  - `days_back`: Number of days to include (default: 30, max: 365)
- **Response**: `WearableSummary` with aggregated metrics

#### Trend Analysis
- **GET** `/api/patients/{patient_id}/wearable/trends`
- Advanced trend analysis for wearable metrics
- **Query Parameters**:
  - `weeks_back`: Number of weeks for analysis (default: 12, max: 52)
  - `metrics`: List of metrics to analyze

#### Recovery Insights
- **GET** `/api/patients/{patient_id}/wearable/recovery-insights`
- AI-enhanced recovery indicators based on wearable data
- **Response**: `RecoveryIndicators`

#### Correlations
- **GET** `/api/patients/{patient_id}/wearable/correlations`
- Analyzes correlations between wearable metrics and PRO scores
- **Query Parameters**:
  - `days_back`: Days of data to analyze (default: 90, max: 365)

### Data Quality & Export

#### Data Quality Report
- **GET** `/api/patients/{patient_id}/wearable/data-quality`
- Comprehensive data quality assessment
- **Response**: `DataQualityReport` with completeness scores and recommendations

#### Data Export
- **GET** `/api/patients/{patient_id}/wearable/export`
- Exports wearable data in various formats for research/insurance
- **Query Parameters**:
  - `format`: Export format (json, csv, fhir)
  - `start_date`: Optional start date
  - `end_date`: Optional end date
  - `include_raw_data`: Include raw data (default: true)
  - `include_aggregated`: Include aggregated data (default: true)
  - `anonymize`: Remove personal identifiers (default: false)

## Data Models

### ComprehensiveWearableData
Comprehensive wearable data record including:
- **Activity Metrics**: Steps, distance, calories, active minutes
- **Heart Rate Metrics**: Resting HR, HRV, recovery metrics
- **Sleep Metrics**: Sleep duration, efficiency, stages
- **Movement Metrics**: Walking speed, gait analysis, mobility scores
- **Physiological Metrics**: Oxygen saturation, blood pressure
- **Environmental Data**: Temperature, humidity, air quality
- **Exercise Sessions**: Detailed workout tracking
- **Sleep Sessions**: Detailed sleep stage analysis

### Data Quality Features
- **Source Tracking**: HealthKit, Google Fit, Fitbit, etc.
- **Quality Indicators**: High, Medium, Low, Poor
- **Confidence Scoring**: 0-1 confidence in data accuracy
- **Gap Detection**: Automatic identification of missing data periods
- **Validation**: Physiological range checking

## Error Handling

### Custom Exceptions
- `WearableAPIError`: Custom exception with status codes and details
- Comprehensive error logging
- Validation error handling
- Database error management

### HTTP Status Codes
- **200**: Success
- **404**: Patient or data not found
- **422**: Validation error
- **500**: Internal server error

## Features

### Advanced Filtering
- Date range filtering with timezone support
- Data source filtering
- Quality threshold filtering
- Pagination with offset/limit

### Mobile Optimization
- Efficient batch operations
- Conflict resolution for offline sync
- Incremental sync tracking
- Optimized payloads

### Analytics Engine
- Weekly/monthly aggregations
- Trend detection algorithms
- Recovery velocity calculations
- Anomaly detection

### Export Capabilities
- Multiple format support (CSV, JSON, FHIR)
- Research-grade data export
- Insurance claim formatting
- Data anonymization

## Integration Notes

### Database Setup
The router requires database initialization:
```python
from routers.wearable_api import set_database
set_database(db)  # MongoDB database instance
```

### Dependencies
- FastAPI
- Motor (MongoDB async driver)
- Pydantic for data validation
- Custom wearable data models
- Schema utilities for aggregations

### Security
- Patient validation on all endpoints
- Input validation and sanitization
- Comprehensive error handling
- Logging for audit trails

## Usage Examples

### Creating Wearable Data
```python
data = {
    "date": "2024-01-15",
    "activity_metrics": {
        "steps": 8500,
        "distance_meters": 6800,
        "active_minutes": 45
    },
    "heart_rate_metrics": {
        "resting_hr": 65,
        "avg_hr": 85,
        "hr_variability_ms": 35
    },
    "sleep_metrics": {
        "total_sleep_minutes": 450,
        "sleep_efficiency": 85.5
    }
}
```

### Bulk Import
```python
import_data = {
    "patient_id": "patient-123",
    "import_source": "HealthKit",
    "data_records": [data1, data2, data3]
}
```

This comprehensive API provides all the necessary functionality for managing wearable data in the RcvryAI orthopedic recovery application, with support for real-time sync, advanced analytics, and data export capabilities.