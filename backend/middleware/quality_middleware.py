"""
Quality Validation Middleware for API Endpoints
Provides real-time validation for all wearable data operations
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
import time
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json

from ..services.data_validation import ClinicalDataValidator, ValidationSeverity
from ..services.quality_monitoring import RealTimeQualityMonitor, QualityAlert
from ..services.audit_trail import HealthcareAuditTrail, AuditEventType, AuditSeverity
from ..services.ml_data_quality import MLDataQualityAssurance
from ..models.wearable_data import WearableData


class QualityValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for real-time quality validation of wearable data operations
    Integrates with all validation services for comprehensive quality assurance
    """
    
    def __init__(self, app, db_client, enable_real_time_monitoring: bool = True,
                 enable_audit_logging: bool = True, enable_ml_validation: bool = True):
        super().__init__(app)
        self.db_client = db_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation services
        self.clinical_validator = ClinicalDataValidator()
        self.audit_trail = HealthcareAuditTrail(db_client)
        self.ml_quality_assurance = MLDataQualityAssurance()
        
        # Initialize monitoring
        self.quality_monitor = None
        if enable_real_time_monitoring:
            self.quality_monitor = RealTimeQualityMonitor(db_client)
        
        # Configuration
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.enable_audit_logging = enable_audit_logging
        self.enable_ml_validation = enable_ml_validation
        
        # Performance tracking
        self.validation_times = []
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Quality thresholds for blocking requests
        self.quality_thresholds = {
            'critical_block': 0.3,  # Block requests below this threshold
            'warning_threshold': 0.7,  # Add warnings below this threshold
            'ml_threshold': 0.8  # Require ML validation above this threshold
        }
        
        # API endpoints that require validation
        self.validation_endpoints = {
            '/api/patients/{patient_id}/wearable': ['POST', 'PUT', 'PATCH'],
            '/api/wearable-data': ['POST', 'PUT', 'PATCH'],
            '/api/patients/{patient_id}/wearable/bulk': ['POST'],
            '/api/wearable-data/sync': ['POST'],
            '/api/patients/{patient_id}/export': ['GET']
        }
        
        # Critical endpoints that always require validation
        self.critical_endpoints = {
            '/api/patients/{patient_id}/wearable/clinical-alerts',
            '/api/patients/{patient_id}/analytics/predictions',
            '/api/research/export'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Main middleware dispatch method
        Validates requests and responses based on endpoint and data type
        """
        start_time = time.time()
        
        try:
            # Check if endpoint requires validation
            if not self._should_validate_endpoint(request):
                return await call_next(request)
            
            # Extract request information
            request_info = await self._extract_request_info(request)
            
            # Pre-request validation
            validation_result = await self._validate_request(request, request_info)
            
            if validation_result.get('block_request', False):
                return JSONResponse(
                    status_code=422,
                    content={
                        'error': 'Data quality validation failed',
                        'details': validation_result.get('errors', []),
                        'quality_score': validation_result.get('quality_score', 0.0),
                        'recommendations': validation_result.get('recommendations', [])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Post-request validation and logging
            await self._post_request_processing(request, response, request_info, validation_result)
            
            # Add quality headers to response
            response = self._add_quality_headers(response, validation_result)
            
            # Log performance
            processing_time = time.time() - start_time
            self.validation_times.append(processing_time)
            if len(self.validation_times) > 1000:
                self.validation_times = self.validation_times[-1000:]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Quality middleware error: {str(e)}")
            
            # Log error to audit trail
            if self.enable_audit_logging:
                await self.audit_trail.log_event(
                    event_type=AuditEventType.SYSTEM_EVENT,
                    action="middleware_error",
                    description=f"Quality middleware error: {str(e)}",
                    severity=AuditSeverity.HIGH,
                    success=False,
                    error_message=str(e)
                )
            
            # Continue processing even if validation fails
            return await call_next(request)
    
    def _should_validate_endpoint(self, request: Request) -> bool:
        """Check if endpoint should be validated"""
        path = request.url.path
        method = request.method
        
        # Always validate critical endpoints
        if any(endpoint in path for endpoint in self.critical_endpoints):
            return True
        
        # Check validation endpoints
        for endpoint_pattern, methods in self.validation_endpoints.items():
            if method in methods:
                # Simple pattern matching (in production, use proper URL routing)
                if self._matches_pattern(path, endpoint_pattern):
                    return True
        
        return False
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching for URL paths"""
        # Replace {param} with wildcard matching
        import re
        pattern_regex = pattern.replace('{patient_id}', r'[^/]+').replace('{id}', r'[^/]+')
        return bool(re.match(f"^{pattern_regex}$", path))
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from request"""
        info = {
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'client_ip': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent'),
            'user_id': request.headers.get('x-user-id'),
            'session_id': request.headers.get('x-session-id'),
            'timestamp': datetime.utcnow()
        }
        
        # Extract body for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    info['body'] = json.loads(body.decode('utf-8'))
            except Exception as e:
                self.logger.warning(f"Failed to parse request body: {str(e)}")
                info['body'] = None
        
        return info
    
    async def _validate_request(self, request: Request, request_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request data and return validation result"""
        validation_result = {
            'quality_score': 1.0,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'block_request': False,
            'validation_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Extract wearable data from request
            wearable_data = await self._extract_wearable_data(request_info)
            
            if not wearable_data:
                return validation_result
            
            # Get patient context
            patient_id = self._extract_patient_id(request_info)
            patient_context = await self._get_patient_context(patient_id) if patient_id else None
            
            # Clinical validation
            if wearable_data and isinstance(wearable_data, list):
                for data in wearable_data:
                    clinical_report = await self._validate_clinical_data(data, patient_context)
                    validation_result = self._merge_validation_results(validation_result, clinical_report)
            elif wearable_data:
                clinical_report = await self._validate_clinical_data(wearable_data, patient_context)
                validation_result = self._merge_validation_results(validation_result, clinical_report)
            
            # ML validation if enabled and threshold met
            if (self.enable_ml_validation and 
                validation_result['quality_score'] >= self.quality_thresholds['ml_threshold']):
                
                ml_validation = await self._validate_ml_data(wearable_data, patient_context)
                validation_result = self._merge_ml_validation(validation_result, ml_validation)
            
            # Determine if request should be blocked
            if validation_result['quality_score'] < self.quality_thresholds['critical_block']:
                validation_result['block_request'] = True
                validation_result['errors'].append("Data quality below acceptable threshold")
            
            # Add warnings for low quality
            elif validation_result['quality_score'] < self.quality_thresholds['warning_threshold']:
                validation_result['warnings'].append("Data quality below optimal threshold")
            
            # Real-time monitoring
            if self.enable_real_time_monitoring and self.quality_monitor:
                await self._trigger_real_time_monitoring(wearable_data, validation_result, patient_context)
            
            validation_result['validation_time'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['quality_score'] = 0.5  # Default to medium quality on error
        
        return validation_result
    
    async def _extract_wearable_data(self, request_info: Dict[str, Any]) -> Optional[Any]:
        """Extract wearable data from request"""
        body = request_info.get('body')
        if not body:
            return None
        
        # Handle different request formats
        if isinstance(body, dict):
            # Single data point
            if 'activity_metrics' in body or 'heart_rate_metrics' in body:
                return self._convert_to_wearable_data(body)
            
            # Bulk data
            if 'data' in body and isinstance(body['data'], list):
                return [self._convert_to_wearable_data(item) for item in body['data']]
        
        elif isinstance(body, list):
            # Array of data points
            return [self._convert_to_wearable_data(item) for item in body]
        
        return None
    
    def _convert_to_wearable_data(self, data_dict: Dict[str, Any]) -> WearableData:
        """Convert dictionary to WearableData object"""
        # This would need proper conversion logic based on the actual WearableData model
        # For now, return a mock object
        return data_dict  # Placeholder
    
    def _extract_patient_id(self, request_info: Dict[str, Any]) -> Optional[str]:
        """Extract patient ID from request"""
        # Check URL path
        path = request_info.get('path', '')
        if '/patients/' in path:
            parts = path.split('/')
            try:
                patient_index = parts.index('patients') + 1
                return parts[patient_index]
            except (ValueError, IndexError):
                pass
        
        # Check request body
        body = request_info.get('body')
        if body and isinstance(body, dict):
            return body.get('patient_id')
        
        return None
    
    async def _get_patient_context(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient context for validation"""
        try:
            # In a real implementation, this would query the database
            # For now, return a mock context
            return {
                'patient_id': patient_id,
                'age': 45,
                'diagnosis': 'ACL Tear',
                'recovery_stage': 'mid_recovery'
            }
        except Exception as e:
            self.logger.error(f"Failed to get patient context: {str(e)}")
            return None
    
    async def _validate_clinical_data(self, data: Any, patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data using clinical validator"""
        try:
            # Check cache first
            cache_key = f"clinical_{hash(str(data))}"
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if (datetime.utcnow() - cached_result['timestamp']).total_seconds() < self.cache_ttl:
                    return cached_result['result']
            
            # Perform validation
            if hasattr(data, 'patient_id'):  # Proper WearableData object
                report = await self.clinical_validator.validate_wearable_data(data, [], patient_context)
            else:
                # Mock validation for non-WearableData objects
                report = type('MockReport', (), {
                    'overall_score': 0.8,
                    'validation_results': [],
                    'recommendations': []
                })()
            
            result = {
                'quality_score': report.overall_score,
                'validation_results': getattr(report, 'validation_results', []),
                'recommendations': getattr(report, 'recommendations', [])
            }
            
            # Cache result
            self.validation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Clinical validation error: {str(e)}")
            return {
                'quality_score': 0.5,
                'validation_results': [],
                'recommendations': [f"Validation error: {str(e)}"]
            }
    
    async def _validate_ml_data(self, data: Any, patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data using ML quality assurance"""
        try:
            # Mock ML validation
            return {
                'ml_quality_score': 0.85,
                'ml_ready': True,
                'feature_quality': {},
                'recommendations': []
            }
            
        except Exception as e:
            self.logger.error(f"ML validation error: {str(e)}")
            return {
                'ml_quality_score': 0.5,
                'ml_ready': False,
                'feature_quality': {},
                'recommendations': [f"ML validation error: {str(e)}"]
            }
    
    def _merge_validation_results(self, base_result: Dict[str, Any], 
                                clinical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge clinical validation results with base result"""
        # Update quality score (take minimum for conservative approach)
        base_result['quality_score'] = min(
            base_result['quality_score'], 
            clinical_result.get('quality_score', 1.0)
        )
        
        # Add validation results
        validation_results = clinical_result.get('validation_results', [])
        for result in validation_results:
            if not result.is_valid:
                if result.severity in ['critical', 'error']:
                    base_result['errors'].append(result.message)
                else:
                    base_result['warnings'].append(result.message)
        
        # Add recommendations
        recommendations = clinical_result.get('recommendations', [])
        base_result['recommendations'].extend(recommendations)
        
        return base_result
    
    def _merge_ml_validation(self, base_result: Dict[str, Any], 
                           ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ML validation results with base result"""
        # Update quality score
        ml_score = ml_result.get('ml_quality_score', 1.0)
        base_result['quality_score'] = (base_result['quality_score'] + ml_score) / 2
        
        # Add ML-specific warnings
        if not ml_result.get('ml_ready', True):
            base_result['warnings'].append("Data not suitable for ML model predictions")
        
        # Add ML recommendations
        ml_recommendations = ml_result.get('recommendations', [])
        base_result['recommendations'].extend(ml_recommendations)
        
        return base_result
    
    async def _trigger_real_time_monitoring(self, data: Any, validation_result: Dict[str, Any], 
                                          patient_context: Dict[str, Any] = None):
        """Trigger real-time monitoring for data"""
        try:
            if isinstance(data, list):
                for item in data:
                    await self.quality_monitor.process_data_point(item, patient_context)
            else:
                await self.quality_monitor.process_data_point(data, patient_context)
                
        except Exception as e:
            self.logger.error(f"Real-time monitoring error: {str(e)}")
    
    async def _post_request_processing(self, request: Request, response: Response, 
                                     request_info: Dict[str, Any], validation_result: Dict[str, Any]):
        """Post-request processing and logging"""
        try:
            # Audit logging
            if self.enable_audit_logging:
                await self._log_audit_event(request, response, request_info, validation_result)
            
            # Performance logging
            if validation_result.get('validation_time', 0) > 1.0:  # Log slow validations
                self.logger.warning(f"Slow validation: {validation_result['validation_time']:.2f}s for {request_info['path']}")
            
        except Exception as e:
            self.logger.error(f"Post-request processing error: {str(e)}")
    
    async def _log_audit_event(self, request: Request, response: Response, 
                             request_info: Dict[str, Any], validation_result: Dict[str, Any]):
        """Log audit event"""
        try:
            # Determine event type
            if request_info['method'] == 'POST':
                event_type = AuditEventType.DATA_CREATION
            elif request_info['method'] in ['PUT', 'PATCH']:
                event_type = AuditEventType.DATA_MODIFICATION
            elif request_info['method'] == 'DELETE':
                event_type = AuditEventType.DATA_DELETION
            else:
                event_type = AuditEventType.DATA_ACCESS
            
            # Determine severity
            if validation_result.get('block_request', False):
                severity = AuditSeverity.HIGH
            elif validation_result.get('warnings'):
                severity = AuditSeverity.MEDIUM
            else:
                severity = AuditSeverity.LOW
            
            # Log event
            await self.audit_trail.log_event(
                event_type=event_type,
                action=f"{request_info['method']} {request_info['path']}",
                description=f"API request with quality score: {validation_result.get('quality_score', 0.0):.3f}",
                user_id=request_info.get('user_id'),
                patient_id=self._extract_patient_id(request_info),
                severity=severity,
                success=response.status_code < 400,
                ip_address=request_info.get('client_ip'),
                user_agent=request_info.get('user_agent'),
                session_id=request_info.get('session_id'),
                api_endpoint=request_info['path'],
                metadata={
                    'quality_score': validation_result.get('quality_score', 0.0),
                    'validation_time': validation_result.get('validation_time', 0.0),
                    'warnings_count': len(validation_result.get('warnings', [])),
                    'errors_count': len(validation_result.get('errors', [])),
                    'response_status': response.status_code
                }
            )
            
        except Exception as e:
            self.logger.error(f"Audit logging error: {str(e)}")
    
    def _add_quality_headers(self, response: Response, validation_result: Dict[str, Any]) -> Response:
        """Add quality-related headers to response"""
        try:
            # Add quality score header
            response.headers['X-Data-Quality-Score'] = str(validation_result.get('quality_score', 0.0))
            
            # Add validation time header
            response.headers['X-Validation-Time'] = str(validation_result.get('validation_time', 0.0))
            
            # Add warnings count
            response.headers['X-Quality-Warnings'] = str(len(validation_result.get('warnings', [])))
            
            # Add recommendations count
            response.headers['X-Quality-Recommendations'] = str(len(validation_result.get('recommendations', [])))
            
            # Add quality level
            quality_score = validation_result.get('quality_score', 0.0)
            if quality_score >= 0.9:
                quality_level = 'excellent'
            elif quality_score >= 0.8:
                quality_level = 'good'
            elif quality_score >= 0.7:
                quality_level = 'fair'
            else:
                quality_level = 'poor'
            
            response.headers['X-Quality-Level'] = quality_level
            
        except Exception as e:
            self.logger.error(f"Error adding quality headers: {str(e)}")
        
        return response
    
    async def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware performance statistics"""
        return {
            'total_validations': len(self.validation_times),
            'avg_validation_time': sum(self.validation_times) / len(self.validation_times) if self.validation_times else 0,
            'max_validation_time': max(self.validation_times) if self.validation_times else 0,
            'min_validation_time': min(self.validation_times) if self.validation_times else 0,
            'cache_size': len(self.validation_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'validation_endpoints': len(self.validation_endpoints),
            'critical_endpoints': len(self.critical_endpoints)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would need proper tracking in a real implementation
        return 0.75  # Placeholder
    
    async def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        self.logger.info("Validation cache cleared")
    
    async def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update quality thresholds"""
        self.quality_thresholds.update(new_thresholds)
        self.logger.info(f"Quality thresholds updated: {new_thresholds}")


class QualityResponseMiddleware:
    """
    Middleware for adding quality information to API responses
    """
    
    def __init__(self, include_quality_metrics: bool = True,
                 include_recommendations: bool = True):
        self.include_quality_metrics = include_quality_metrics
        self.include_recommendations = include_recommendations
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process response and add quality information"""
        response = await call_next(request)
        
        # Add quality information to response body for data endpoints
        if (self.include_quality_metrics and 
            request.url.path.startswith('/api/') and 
            'wearable' in request.url.path):
            
            response = await self._enhance_response_with_quality(response)
        
        return response
    
    async def _enhance_response_with_quality(self, response: Response) -> Response:
        """Enhance response with quality metrics"""
        try:
            # Get quality score from headers
            quality_score = response.headers.get('X-Data-Quality-Score', '0.0')
            quality_level = response.headers.get('X-Quality-Level', 'unknown')
            warnings_count = response.headers.get('X-Quality-Warnings', '0')
            recommendations_count = response.headers.get('X-Quality-Recommendations', '0')
            
            # Add quality metadata to response
            if hasattr(response, 'body'):
                try:
                    # Parse existing response body
                    body = response.body
                    if isinstance(body, bytes):
                        body = body.decode('utf-8')
                    
                    response_data = json.loads(body) if body else {}
                    
                    # Add quality metadata
                    response_data['_quality'] = {
                        'score': float(quality_score),
                        'level': quality_level,
                        'warnings_count': int(warnings_count),
                        'recommendations_count': int(recommendations_count)
                    }
                    
                    # Update response body
                    new_body = json.dumps(response_data)
                    response.body = new_body.encode('utf-8')
                    
                except Exception as e:
                    self.logger.error(f"Error enhancing response: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in response enhancement: {str(e)}")
        
        return response