# Comprehensive Wearable Data Implementation

## Overview
Enhanced wearable data storage and analytics for orthopedic recovery tracking, supporting HealthKit, Android Health, and other wearable device integrations.

## New Components

### 1. Models (`models/wearable_data.py`)
- **ComprehensiveWearableData**: Main model with rich health metrics
- **ActivityMetrics**: Steps, distance, calories, active/sedentary time
- **HeartRateMetrics**: Resting HR, HRV, recovery metrics, zones
- **SleepMetrics**: Duration, efficiency, stages, quality scores
- **MovementMetrics**: Walking speed, gait analysis, mobility scores
- **PhysiologicalMetrics**: O2 saturation, blood pressure, temperature
- **DataMetadata**: Source tracking, quality metrics, confidence scores

### 2. MongoDB Schemas (`schemas/wearable_schemas.py`)
- **Optimized indexes** for fast queries by patient, date, metrics
- **Validation schemas** with physiological range checking
- **Aggregation pipelines** for trends, correlations, analytics
- **Data quality monitoring** and gap detection utilities

### 3. Enhanced API Endpoints

#### Comprehensive Data Management
- `POST /api/wearable-data/comprehensive` - Create rich wearable data
- `GET /api/wearable-data/comprehensive/{patient_id}` - Retrieve data
- `POST /api/wearable-data/bulk-import` - Bulk import from external sources

#### Analytics and Insights
- `GET /api/wearable-data/trends/{patient_id}` - Weekly trends analysis
- `GET /api/wearable-data/recovery-indicators/{patient_id}` - Recovery alerts
- `GET /api/wearable-data/data-quality/{patient_id}` - Quality metrics

### 4. Enhanced AI Insights
Advanced analytics that detect:
- **Walking speed trends** - Declining mobility patterns
- **Activity drops** - Significant decreases preceding pain increases  
- **Sleep quality impact** - Correlation with recovery metrics
- **Sedentary time alerts** - Extended inactivity warnings
- **Heart rate variability** - Recovery stress indicators
- **PRO score correlations** - Activity/sleep vs clinical outcomes

## Key Features

### Data Sources Supported
- HealthKit (iOS)
- Google Fit (Android)
- Fitbit
- Samsung Health
- Garmin
- Manual entry

### Advanced Analytics
- **Trend detection** with statistical significance
- **Pattern recognition** for activity/sleep cycles
- **Risk assessment** based on multiple metrics
- **Correlation analysis** with PRO questionnaires
- **Milestone tracking** against clinical recovery timelines

### Data Quality Management
- **Source tracking** and confidence scoring
- **Validation** with physiological ranges
- **Gap detection** for missing data periods
- **Quality metrics** and completeness reporting

## Usage Examples

### Creating Comprehensive Data
```python
from models.wearable_data import ComprehensiveWearableDataCreate, ActivityMetrics

data = ComprehensiveWearableDataCreate(
    patient_id="patient-123",
    date=date.today(),
    activity_metrics=ActivityMetrics(
        steps=8000,
        distance_meters=5000,
        active_minutes=120
    ),
    heart_rate_metrics=HeartRateMetrics(
        resting_hr=65,
        average_hr=85
    )
)
```

### Bulk Import from HealthKit
```python
bulk_import = BulkWearableDataImport(
    patient_id="patient-123",
    data_records=[...],  # List of daily records
    import_source=DataSource.HEALTHKIT
)
```

### Analyzing Recovery Indicators
The system automatically detects:
- Walking speed declining >10% over 2 weeks
- Activity drops >30% preceding pain increases
- Sleep efficiency below 75% affecting recovery
- Sedentary time exceeding 10 hours daily

## Database Collections

### Primary Collections
- `comprehensive_wearable_data` - Main wearable data storage
- `recovery_indicators` - Analysis results and alerts
- `wearable_data_trends` - Computed trend analytics

### Legacy Compatibility
- Maintains `wearable_data` collection for backward compatibility
- Automatic fallback to legacy data when comprehensive data unavailable

## Performance Optimizations
- **Compound indexes** on patient_id + date for fast filtering
- **Aggregation pipelines** for complex analytics without application logic
- **Batch operations** for bulk imports
- **Data validation** at the database level

## Future Enhancements
- Real-time data streaming via WebSocket endpoints
- Machine learning models for predictive analytics
- Integration with Capacitor.js for mobile apps
- Advanced visualization dashboards
- Clinical decision support alerts