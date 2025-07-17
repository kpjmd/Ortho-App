"""
Healthcare Compliance and Audit Trail System
Provides HIPAA-compliant audit trails and FDA-ready documentation
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from motor.motor_asyncio import AsyncIOMotorClient

from ..models.wearable_data import WearableData


class AuditEventType(str, Enum):
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_CREATION = "data_creation"
    DATA_DELETION = "data_deletion"
    EXPORT = "export"
    VALIDATION = "validation"
    QUALITY_CHECK = "quality_check"
    ANONYMIZATION = "anonymization"
    CONSENT_CHANGE = "consent_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    CLINICAL_ALERT = "clinical_alert"
    ML_PREDICTION = "ml_prediction"
    RESEARCH_ACCESS = "research_access"


class ComplianceLevel(str, Enum):
    HIPAA = "hipaa"
    FDA_21CFR11 = "fda_21cfr11"
    GCP = "gcp"
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"


class AuditSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    
    # User and system information
    user_id: Optional[str]
    user_role: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    
    # Data information
    patient_id: Optional[str]
    data_type: Optional[str]
    data_id: Optional[str]
    
    # Event details
    action: str
    description: str
    before_value: Optional[Dict[str, Any]]
    after_value: Optional[Dict[str, Any]]
    
    # System information
    system_component: str
    api_endpoint: Optional[str]
    request_id: Optional[str]
    
    # Compliance information
    compliance_level: ComplianceLevel
    retention_period: int  # days
    
    # Security
    checksum: str
    encrypted_data: Optional[str]
    
    # Status
    success: bool
    error_message: Optional[str]
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance audit report"""
    report_id: str
    generation_date: datetime
    compliance_level: ComplianceLevel
    audit_period: Tuple[datetime, datetime]
    
    # Summary statistics
    total_events: int
    events_by_type: Dict[AuditEventType, int]
    events_by_severity: Dict[AuditSeverity, int]
    unique_users: int
    unique_patients: int
    
    # Compliance metrics
    data_integrity_score: float
    access_control_compliance: float
    audit_trail_completeness: float
    security_event_rate: float
    
    # Findings
    compliance_violations: List[Dict[str, Any]]
    security_incidents: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Certification
    compliance_status: str
    certifying_authority: str
    next_audit_date: datetime


class HealthcareAuditTrail:
    """
    Healthcare compliance and audit trail system
    Provides comprehensive audit logging for HIPAA, FDA, and other regulatory requirements
    """
    
    def __init__(self, db_client: AsyncIOMotorClient, encryption_key: bytes = None):
        self.db_client = db_client
        self.db = db_client.ortho_app
        self.logger = logging.getLogger(__name__)
        
        # Encryption for sensitive data
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())
        
        # Compliance requirements
        self.compliance_requirements = {
            ComplianceLevel.HIPAA: {
                'retention_period': 2555,  # 7 years
                'required_fields': ['user_id', 'patient_id', 'action', 'timestamp'],
                'encryption_required': True,
                'integrity_checks': True,
                'access_controls': True
            },
            ComplianceLevel.FDA_21CFR11: {
                'retention_period': 1095,  # 3 years minimum
                'required_fields': ['user_id', 'timestamp', 'action', 'before_value', 'after_value'],
                'encryption_required': True,
                'integrity_checks': True,
                'electronic_signatures': True
            },
            ComplianceLevel.GCP: {
                'retention_period': 730,  # 2 years
                'required_fields': ['user_id', 'timestamp', 'action', 'patient_id'],
                'encryption_required': True,
                'integrity_checks': True,
                'monitoring_required': True
            },
            ComplianceLevel.GDPR: {
                'retention_period': 1095,  # 3 years
                'required_fields': ['user_id', 'timestamp', 'action', 'patient_id'],
                'encryption_required': True,
                'right_to_erasure': True,
                'consent_tracking': True
            }
        }
        
        # Initialize collections
        self.audit_collection = self.db.audit_trail
        self.compliance_reports_collection = self.db.compliance_reports
        
        # Performance tracking
        self.audit_buffer = []
        self.buffer_size = 100
        self.last_flush = datetime.utcnow()
    
    async def log_event(self, event_type: AuditEventType, 
                       action: str,
                       description: str,
                       user_id: str = None,
                       patient_id: str = None,
                       data_type: str = None,
                       data_id: str = None,
                       before_value: Dict[str, Any] = None,
                       after_value: Dict[str, Any] = None,
                       severity: AuditSeverity = AuditSeverity.LOW,
                       compliance_level: ComplianceLevel = ComplianceLevel.HIPAA,
                       system_component: str = "wearable_data_system",
                       api_endpoint: str = None,
                       request_id: str = None,
                       user_role: str = None,
                       session_id: str = None,
                       ip_address: str = None,
                       user_agent: str = None,
                       success: bool = True,
                       error_message: str = None,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of audit event
            action: Action performed
            description: Description of the event
            user_id: User who performed the action
            patient_id: Patient ID if applicable
            data_type: Type of data involved
            data_id: ID of specific data record
            before_value: Data before modification
            after_value: Data after modification
            severity: Severity level
            compliance_level: Compliance standard
            system_component: System component involved
            api_endpoint: API endpoint if applicable
            request_id: Request ID for tracing
            user_role: User role
            session_id: Session ID
            ip_address: IP address
            user_agent: User agent
            success: Whether action was successful
            error_message: Error message if failed
            metadata: Additional metadata
            
        Returns:
            Event ID of the logged event
        """
        
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Get retention period based on compliance level
            retention_period = self.compliance_requirements[compliance_level]['retention_period']
            
            # Encrypt sensitive data if required
            encrypted_data = None
            if self.compliance_requirements[compliance_level].get('encryption_required', False):
                sensitive_data = {
                    'before_value': before_value,
                    'after_value': after_value,
                    'metadata': metadata
                }
                encrypted_data = self.cipher.encrypt(json.dumps(sensitive_data, default=str).encode()).decode()
            
            # Calculate checksum for integrity
            checksum_data = {
                'event_id': event_id,
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'action': action,
                'user_id': user_id,
                'patient_id': patient_id
            }
            checksum = hashlib.sha256(json.dumps(checksum_data, sort_keys=True).encode()).hexdigest()
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                user_role=user_role,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                patient_id=patient_id,
                data_type=data_type,
                data_id=data_id,
                action=action,
                description=description,
                before_value=before_value if not encrypted_data else None,
                after_value=after_value if not encrypted_data else None,
                system_component=system_component,
                api_endpoint=api_endpoint,
                request_id=request_id,
                compliance_level=compliance_level,
                retention_period=retention_period,
                checksum=checksum,
                encrypted_data=encrypted_data,
                success=success,
                error_message=error_message,
                metadata=metadata if not encrypted_data else None
            )
            
            # Add to buffer for batch processing
            self.audit_buffer.append(audit_event)
            
            # Flush buffer if full or time-based
            if len(self.audit_buffer) >= self.buffer_size or \
               (datetime.utcnow() - self.last_flush).total_seconds() > 30:
                await self._flush_audit_buffer()
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {str(e)}")
            raise
    
    async def _flush_audit_buffer(self):
        """Flush audit buffer to database"""
        if not self.audit_buffer:
            return
        
        try:
            # Convert to documents
            documents = []
            for event in self.audit_buffer:
                doc = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'user_id': event.user_id,
                    'user_role': event.user_role,
                    'session_id': event.session_id,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent,
                    'patient_id': event.patient_id,
                    'data_type': event.data_type,
                    'data_id': event.data_id,
                    'action': event.action,
                    'description': event.description,
                    'before_value': event.before_value,
                    'after_value': event.after_value,
                    'system_component': event.system_component,
                    'api_endpoint': event.api_endpoint,
                    'request_id': event.request_id,
                    'compliance_level': event.compliance_level,
                    'retention_period': event.retention_period,
                    'checksum': event.checksum,
                    'encrypted_data': event.encrypted_data,
                    'success': event.success,
                    'error_message': event.error_message,
                    'metadata': event.metadata
                }
                documents.append(doc)
            
            # Insert to database
            await self.audit_collection.insert_many(documents)
            
            # Clear buffer
            self.audit_buffer = []
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit buffer: {str(e)}")
            raise
    
    async def log_data_access(self, user_id: str, patient_id: str, data_type: str,
                            action: str = "read", **kwargs) -> str:
        """Log data access event"""
        return await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=action,
            description=f"User {user_id} accessed {data_type} for patient {patient_id}",
            user_id=user_id,
            patient_id=patient_id,
            data_type=data_type,
            severity=AuditSeverity.LOW,
            **kwargs
        )
    
    async def log_data_modification(self, user_id: str, patient_id: str, data_type: str,
                                  data_id: str, before_value: Dict[str, Any], 
                                  after_value: Dict[str, Any], **kwargs) -> str:
        """Log data modification event"""
        return await self.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            action="update",
            description=f"User {user_id} modified {data_type} for patient {patient_id}",
            user_id=user_id,
            patient_id=patient_id,
            data_type=data_type,
            data_id=data_id,
            before_value=before_value,
            after_value=after_value,
            severity=AuditSeverity.MEDIUM,
            **kwargs
        )
    
    async def log_data_creation(self, user_id: str, patient_id: str, data_type: str,
                              data_id: str, data_value: Dict[str, Any], **kwargs) -> str:
        """Log data creation event"""
        return await self.log_event(
            event_type=AuditEventType.DATA_CREATION,
            action="create",
            description=f"User {user_id} created {data_type} for patient {patient_id}",
            user_id=user_id,
            patient_id=patient_id,
            data_type=data_type,
            data_id=data_id,
            after_value=data_value,
            severity=AuditSeverity.LOW,
            **kwargs
        )
    
    async def log_data_deletion(self, user_id: str, patient_id: str, data_type: str,
                              data_id: str, data_value: Dict[str, Any], **kwargs) -> str:
        """Log data deletion event"""
        return await self.log_event(
            event_type=AuditEventType.DATA_DELETION,
            action="delete",
            description=f"User {user_id} deleted {data_type} for patient {patient_id}",
            user_id=user_id,
            patient_id=patient_id,
            data_type=data_type,
            data_id=data_id,
            before_value=data_value,
            severity=AuditSeverity.HIGH,
            **kwargs
        )
    
    async def log_export_event(self, user_id: str, patient_ids: List[str], 
                             export_type: str, **kwargs) -> str:
        """Log data export event"""
        return await self.log_event(
            event_type=AuditEventType.EXPORT,
            action="export",
            description=f"User {user_id} exported {export_type} data for {len(patient_ids)} patients",
            user_id=user_id,
            data_type=export_type,
            severity=AuditSeverity.HIGH,
            metadata={"patient_count": len(patient_ids), "export_type": export_type},
            **kwargs
        )
    
    async def log_validation_event(self, user_id: str, patient_id: str, 
                                 validation_type: str, result: Dict[str, Any], **kwargs) -> str:
        """Log validation event"""
        return await self.log_event(
            event_type=AuditEventType.VALIDATION,
            action="validate",
            description=f"Data validation ({validation_type}) performed for patient {patient_id}",
            user_id=user_id,
            patient_id=patient_id,
            data_type="validation_result",
            after_value=result,
            severity=AuditSeverity.LOW,
            **kwargs
        )
    
    async def log_security_event(self, event_description: str, user_id: str = None,
                               ip_address: str = None, severity: AuditSeverity = AuditSeverity.HIGH,
                               **kwargs) -> str:
        """Log security event"""
        return await self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            action="security_event",
            description=event_description,
            user_id=user_id,
            ip_address=ip_address,
            severity=severity,
            **kwargs
        )
    
    async def log_clinical_alert(self, patient_id: str, alert_type: str, 
                               alert_data: Dict[str, Any], **kwargs) -> str:
        """Log clinical alert event"""
        return await self.log_event(
            event_type=AuditEventType.CLINICAL_ALERT,
            action="clinical_alert",
            description=f"Clinical alert ({alert_type}) generated for patient {patient_id}",
            patient_id=patient_id,
            data_type="clinical_alert",
            after_value=alert_data,
            severity=AuditSeverity.HIGH,
            **kwargs
        )
    
    async def log_ml_prediction(self, patient_id: str, model_type: str, 
                              prediction_result: Dict[str, Any], **kwargs) -> str:
        """Log ML prediction event"""
        return await self.log_event(
            event_type=AuditEventType.ML_PREDICTION,
            action="ml_prediction",
            description=f"ML prediction ({model_type}) generated for patient {patient_id}",
            patient_id=patient_id,
            data_type="ml_prediction",
            after_value=prediction_result,
            severity=AuditSeverity.MEDIUM,
            **kwargs
        )
    
    async def get_audit_trail(self, patient_id: str = None, user_id: str = None,
                            start_date: datetime = None, end_date: datetime = None,
                            event_type: AuditEventType = None,
                            limit: int = 100) -> List[AuditEvent]:
        """
        Retrieve audit trail with filtering
        
        Args:
            patient_id: Filter by patient ID
            user_id: Filter by user ID
            start_date: Start date for filtering
            end_date: End date for filtering
            event_type: Filter by event type
            limit: Maximum number of records to return
            
        Returns:
            List of audit events
        """
        try:
            # Build query
            query = {}
            
            if patient_id:
                query['patient_id'] = patient_id
            
            if user_id:
                query['user_id'] = user_id
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            if event_type:
                query['event_type'] = event_type
            
            # Execute query
            cursor = self.audit_collection.find(query).sort('timestamp', -1).limit(limit)
            
            # Convert to AuditEvent objects
            events = []
            async for doc in cursor:
                # Decrypt sensitive data if needed
                before_value = doc.get('before_value')
                after_value = doc.get('after_value')
                metadata = doc.get('metadata')
                
                if doc.get('encrypted_data'):
                    try:
                        decrypted_data = json.loads(self.cipher.decrypt(doc['encrypted_data'].encode()).decode())
                        before_value = decrypted_data.get('before_value')
                        after_value = decrypted_data.get('after_value')
                        metadata = decrypted_data.get('metadata')
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt audit data: {str(e)}")
                
                event = AuditEvent(
                    event_id=doc['event_id'],
                    timestamp=doc['timestamp'],
                    event_type=doc['event_type'],
                    severity=doc['severity'],
                    user_id=doc.get('user_id'),
                    user_role=doc.get('user_role'),
                    session_id=doc.get('session_id'),
                    ip_address=doc.get('ip_address'),
                    user_agent=doc.get('user_agent'),
                    patient_id=doc.get('patient_id'),
                    data_type=doc.get('data_type'),
                    data_id=doc.get('data_id'),
                    action=doc['action'],
                    description=doc['description'],
                    before_value=before_value,
                    after_value=after_value,
                    system_component=doc['system_component'],
                    api_endpoint=doc.get('api_endpoint'),
                    request_id=doc.get('request_id'),
                    compliance_level=doc['compliance_level'],
                    retention_period=doc['retention_period'],
                    checksum=doc['checksum'],
                    encrypted_data=doc.get('encrypted_data'),
                    success=doc['success'],
                    error_message=doc.get('error_message'),
                    metadata=metadata or {}
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit trail: {str(e)}")
            raise
    
    async def verify_audit_integrity(self, event_id: str = None, 
                                   start_date: datetime = None,
                                   end_date: datetime = None) -> Dict[str, Any]:
        """
        Verify audit trail integrity
        
        Args:
            event_id: Specific event to verify
            start_date: Start date for verification
            end_date: End date for verification
            
        Returns:
            Integrity verification results
        """
        try:
            # Build query
            query = {}
            if event_id:
                query['event_id'] = event_id
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Get events
            cursor = self.audit_collection.find(query)
            
            verification_results = {
                'total_events': 0,
                'verified_events': 0,
                'failed_events': 0,
                'integrity_score': 0.0,
                'failed_checksums': []
            }
            
            async for doc in cursor:
                verification_results['total_events'] += 1
                
                # Verify checksum
                checksum_data = {
                    'event_id': doc['event_id'],
                    'timestamp': doc['timestamp'].isoformat(),
                    'event_type': doc['event_type'],
                    'action': doc['action'],
                    'user_id': doc.get('user_id'),
                    'patient_id': doc.get('patient_id')
                }
                
                calculated_checksum = hashlib.sha256(
                    json.dumps(checksum_data, sort_keys=True).encode()
                ).hexdigest()
                
                if calculated_checksum == doc['checksum']:
                    verification_results['verified_events'] += 1
                else:
                    verification_results['failed_events'] += 1
                    verification_results['failed_checksums'].append({
                        'event_id': doc['event_id'],
                        'timestamp': doc['timestamp'],
                        'expected': doc['checksum'],
                        'calculated': calculated_checksum
                    })
            
            # Calculate integrity score
            if verification_results['total_events'] > 0:
                verification_results['integrity_score'] = (
                    verification_results['verified_events'] / verification_results['total_events']
                )
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Failed to verify audit integrity: {str(e)}")
            raise
    
    async def generate_compliance_report(self, compliance_level: ComplianceLevel,
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> ComplianceReport:
        """
        Generate compliance report
        
        Args:
            compliance_level: Compliance standard to report on
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Compliance report
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Build query
            query = {
                'compliance_level': compliance_level,
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            # Get events
            cursor = self.audit_collection.find(query)
            
            # Initialize report data
            total_events = 0
            events_by_type = {}
            events_by_severity = {}
            unique_users = set()
            unique_patients = set()
            compliance_violations = []
            security_incidents = []
            
            async for doc in cursor:
                total_events += 1
                
                # Count by type
                event_type = doc['event_type']
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                
                # Count by severity
                severity = doc['severity']
                events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
                
                # Track unique users and patients
                if doc.get('user_id'):
                    unique_users.add(doc['user_id'])
                if doc.get('patient_id'):
                    unique_patients.add(doc['patient_id'])
                
                # Check for compliance violations
                if not doc['success'] and doc['severity'] in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                    compliance_violations.append({
                        'event_id': doc['event_id'],
                        'timestamp': doc['timestamp'],
                        'description': doc['description'],
                        'severity': doc['severity'],
                        'error_message': doc.get('error_message')
                    })
                
                # Check for security incidents
                if doc['event_type'] == AuditEventType.SECURITY_EVENT:
                    security_incidents.append({
                        'event_id': doc['event_id'],
                        'timestamp': doc['timestamp'],
                        'description': doc['description'],
                        'severity': doc['severity'],
                        'user_id': doc.get('user_id'),
                        'ip_address': doc.get('ip_address')
                    })
            
            # Calculate compliance metrics
            data_integrity_score = await self._calculate_data_integrity_score(start_date, end_date)
            access_control_compliance = await self._calculate_access_control_compliance(start_date, end_date)
            audit_trail_completeness = await self._calculate_audit_trail_completeness(start_date, end_date)
            security_event_rate = len(security_incidents) / max(total_events, 1)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                compliance_violations, security_incidents, total_events
            )
            
            # Determine compliance status
            compliance_status = self._determine_compliance_status(
                data_integrity_score, access_control_compliance, 
                audit_trail_completeness, len(compliance_violations)
            )
            
            # Generate report
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                generation_date=datetime.utcnow(),
                compliance_level=compliance_level,
                audit_period=(start_date, end_date),
                total_events=total_events,
                events_by_type=events_by_type,
                events_by_severity=events_by_severity,
                unique_users=len(unique_users),
                unique_patients=len(unique_patients),
                data_integrity_score=data_integrity_score,
                access_control_compliance=access_control_compliance,
                audit_trail_completeness=audit_trail_completeness,
                security_event_rate=security_event_rate,
                compliance_violations=compliance_violations,
                security_incidents=security_incidents,
                recommendations=recommendations,
                compliance_status=compliance_status,
                certifying_authority="RcvryAI Compliance Authority",
                next_audit_date=datetime.utcnow() + timedelta(days=90)
            )
            
            # Store report
            await self.compliance_reports_collection.insert_one({
                'report_id': report.report_id,
                'generation_date': report.generation_date,
                'compliance_level': report.compliance_level,
                'audit_period': report.audit_period,
                'total_events': report.total_events,
                'events_by_type': report.events_by_type,
                'events_by_severity': report.events_by_severity,
                'unique_users': report.unique_users,
                'unique_patients': report.unique_patients,
                'data_integrity_score': report.data_integrity_score,
                'access_control_compliance': report.access_control_compliance,
                'audit_trail_completeness': report.audit_trail_completeness,
                'security_event_rate': report.security_event_rate,
                'compliance_violations': report.compliance_violations,
                'security_incidents': report.security_incidents,
                'recommendations': report.recommendations,
                'compliance_status': report.compliance_status,
                'certifying_authority': report.certifying_authority,
                'next_audit_date': report.next_audit_date
            })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {str(e)}")
            raise
    
    async def _calculate_data_integrity_score(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate data integrity score"""
        integrity_results = await self.verify_audit_integrity(start_date=start_date, end_date=end_date)
        return integrity_results['integrity_score']
    
    async def _calculate_access_control_compliance(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate access control compliance score"""
        # Count authorized vs unauthorized access attempts
        query = {
            'event_type': AuditEventType.DATA_ACCESS,
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }
        
        total_access = await self.audit_collection.count_documents(query)
        unauthorized_access = await self.audit_collection.count_documents({
            **query,
            'success': False
        })
        
        if total_access == 0:
            return 1.0
        
        return 1.0 - (unauthorized_access / total_access)
    
    async def _calculate_audit_trail_completeness(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate audit trail completeness score"""
        # Check if all required fields are present
        query = {
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }
        
        total_events = await self.audit_collection.count_documents(query)
        
        if total_events == 0:
            return 1.0
        
        # Count events with all required fields
        complete_events = await self.audit_collection.count_documents({
            **query,
            'user_id': {'$exists': True, '$ne': None},
            'action': {'$exists': True, '$ne': None},
            'timestamp': {'$exists': True, '$ne': None}
        })
        
        return complete_events / total_events
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]], 
                                           security_incidents: List[Dict[str, Any]], 
                                           total_events: int) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if len(violations) > 0:
            recommendations.append(f"Address {len(violations)} compliance violations")
        
        if len(security_incidents) > 0:
            recommendations.append(f"Investigate {len(security_incidents)} security incidents")
        
        if total_events < 100:
            recommendations.append("Increase audit logging coverage")
        
        return recommendations
    
    def _determine_compliance_status(self, data_integrity: float, access_control: float,
                                   audit_completeness: float, violations_count: int) -> str:
        """Determine overall compliance status"""
        if (data_integrity >= 0.98 and access_control >= 0.95 and 
            audit_completeness >= 0.98 and violations_count == 0):
            return "COMPLIANT"
        elif (data_integrity >= 0.90 and access_control >= 0.85 and 
              audit_completeness >= 0.90 and violations_count <= 2):
            return "MOSTLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"
    
    async def cleanup_expired_records(self) -> int:
        """Clean up expired audit records based on retention periods"""
        try:
            deleted_count = 0
            
            # Get all compliance levels and their retention periods
            for compliance_level, requirements in self.compliance_requirements.items():
                retention_period = requirements['retention_period']
                cutoff_date = datetime.utcnow() - timedelta(days=retention_period)
                
                # Delete expired records
                result = await self.audit_collection.delete_many({
                    'compliance_level': compliance_level,
                    'timestamp': {'$lt': cutoff_date}
                })
                
                deleted_count += result.deleted_count
            
            self.logger.info(f"Cleaned up {deleted_count} expired audit records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired records: {str(e)}")
            raise