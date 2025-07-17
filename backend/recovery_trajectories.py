"""
Recovery trajectory baselines for orthopedic conditions.
Defines expected recovery progressions based on clinical literature.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

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

class RecoveryMilestone:
    """Represents a key recovery milestone"""
    def __init__(self, week: int, description: str, expected_score: float, 
                 critical: bool = False, subscale: str = "total"):
        self.week = week
        self.description = description
        self.expected_score = expected_score
        self.critical = critical
        self.subscale = subscale

class TrajectoryPoint:
    """Represents a point on the recovery trajectory"""
    def __init__(self, week: int, expected_score: float, 
                 lower_bound: float, upper_bound: float):
        self.week = week
        self.expected_score = expected_score
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

# KOOS Recovery Trajectories (0-100 scale, 100 = no problems)
KOOS_TRAJECTORIES = {
    DiagnosisType.ACL_TEAR: {
        "symptoms": [
            TrajectoryPoint(1, 40, 25, 55),
            TrajectoryPoint(2, 45, 30, 60),
            TrajectoryPoint(4, 60, 45, 75),
            TrajectoryPoint(6, 70, 55, 85),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 80, 65, 95),
            TrajectoryPoint(16, 85, 70, 100),
            TrajectoryPoint(20, 88, 75, 100),
            TrajectoryPoint(24, 90, 80, 100),
        ],
        "pain": [
            TrajectoryPoint(1, 50, 35, 65),
            TrajectoryPoint(2, 55, 40, 70),
            TrajectoryPoint(4, 65, 50, 80),
            TrajectoryPoint(6, 75, 60, 90),
            TrajectoryPoint(8, 80, 65, 95),
            TrajectoryPoint(12, 85, 70, 100),
            TrajectoryPoint(16, 88, 75, 100),
            TrajectoryPoint(20, 90, 80, 100),
            TrajectoryPoint(24, 92, 85, 100),
        ],
        "adl": [
            TrajectoryPoint(1, 45, 30, 60),
            TrajectoryPoint(2, 50, 35, 65),
            TrajectoryPoint(4, 55, 40, 70),
            TrajectoryPoint(6, 65, 50, 80),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 82, 67, 97),
            TrajectoryPoint(16, 88, 73, 100),
            TrajectoryPoint(20, 92, 80, 100),
            TrajectoryPoint(24, 95, 85, 100),
        ],
        "sport": [
            TrajectoryPoint(1, 10, 0, 25),
            TrajectoryPoint(2, 15, 0, 30),
            TrajectoryPoint(4, 20, 5, 35),
            TrajectoryPoint(6, 25, 10, 40),
            TrajectoryPoint(8, 35, 20, 50),
            TrajectoryPoint(12, 50, 35, 65),
            TrajectoryPoint(16, 65, 50, 80),
            TrajectoryPoint(20, 75, 60, 90),
            TrajectoryPoint(24, 80, 65, 95),
        ],
        "qol": [
            TrajectoryPoint(1, 35, 20, 50),
            TrajectoryPoint(2, 40, 25, 55),
            TrajectoryPoint(4, 45, 30, 60),
            TrajectoryPoint(6, 55, 40, 70),
            TrajectoryPoint(8, 65, 50, 80),
            TrajectoryPoint(12, 75, 60, 90),
            TrajectoryPoint(16, 82, 67, 97),
            TrajectoryPoint(20, 88, 73, 100),
            TrajectoryPoint(24, 92, 80, 100),
        ],
    },
    DiagnosisType.MENISCUS_TEAR: {
        "symptoms": [
            TrajectoryPoint(1, 50, 35, 65),
            TrajectoryPoint(2, 60, 45, 75),
            TrajectoryPoint(4, 70, 55, 85),
            TrajectoryPoint(6, 80, 65, 95),
            TrajectoryPoint(8, 85, 70, 100),
            TrajectoryPoint(12, 90, 75, 100),
            TrajectoryPoint(16, 92, 80, 100),
        ],
        "pain": [
            TrajectoryPoint(1, 55, 40, 70),
            TrajectoryPoint(2, 65, 50, 80),
            TrajectoryPoint(4, 75, 60, 90),
            TrajectoryPoint(6, 82, 67, 97),
            TrajectoryPoint(8, 88, 73, 100),
            TrajectoryPoint(12, 92, 80, 100),
            TrajectoryPoint(16, 95, 85, 100),
        ],
        "adl": [
            TrajectoryPoint(1, 50, 35, 65),
            TrajectoryPoint(2, 60, 45, 75),
            TrajectoryPoint(4, 70, 55, 85),
            TrajectoryPoint(6, 80, 65, 95),
            TrajectoryPoint(8, 85, 70, 100),
            TrajectoryPoint(12, 90, 75, 100),
            TrajectoryPoint(16, 95, 85, 100),
        ],
        "sport": [
            TrajectoryPoint(1, 25, 10, 40),
            TrajectoryPoint(2, 35, 20, 50),
            TrajectoryPoint(4, 50, 35, 65),
            TrajectoryPoint(6, 65, 50, 80),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 85, 70, 100),
            TrajectoryPoint(16, 90, 75, 100),
        ],
        "qol": [
            TrajectoryPoint(1, 45, 30, 60),
            TrajectoryPoint(2, 55, 40, 70),
            TrajectoryPoint(4, 65, 50, 80),
            TrajectoryPoint(6, 75, 60, 90),
            TrajectoryPoint(8, 82, 67, 97),
            TrajectoryPoint(12, 88, 73, 100),
            TrajectoryPoint(16, 92, 80, 100),
        ],
    },
    DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT: {
        "symptoms": [
            TrajectoryPoint(1, 30, 15, 45),
            TrajectoryPoint(2, 40, 25, 55),
            TrajectoryPoint(4, 55, 40, 70),
            TrajectoryPoint(6, 65, 50, 80),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 82, 67, 97),
            TrajectoryPoint(16, 88, 73, 100),
            TrajectoryPoint(20, 92, 80, 100),
            TrajectoryPoint(24, 95, 85, 100),
        ],
        "pain": [
            TrajectoryPoint(1, 35, 20, 50),
            TrajectoryPoint(2, 45, 30, 60),
            TrajectoryPoint(4, 60, 45, 75),
            TrajectoryPoint(6, 70, 55, 85),
            TrajectoryPoint(8, 80, 65, 95),
            TrajectoryPoint(12, 85, 70, 100),
            TrajectoryPoint(16, 90, 75, 100),
            TrajectoryPoint(20, 92, 80, 100),
            TrajectoryPoint(24, 95, 85, 100),
        ],
        "adl": [
            TrajectoryPoint(1, 25, 10, 40),
            TrajectoryPoint(2, 35, 20, 50),
            TrajectoryPoint(4, 50, 35, 65),
            TrajectoryPoint(6, 65, 50, 80),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 82, 67, 97),
            TrajectoryPoint(16, 88, 73, 100),
            TrajectoryPoint(20, 92, 80, 100),
            TrajectoryPoint(24, 95, 85, 100),
        ],
        "sport": [
            TrajectoryPoint(1, 5, 0, 15),
            TrajectoryPoint(2, 10, 0, 20),
            TrajectoryPoint(4, 15, 5, 25),
            TrajectoryPoint(6, 25, 10, 40),
            TrajectoryPoint(8, 35, 20, 50),
            TrajectoryPoint(12, 50, 35, 65),
            TrajectoryPoint(16, 60, 45, 75),
            TrajectoryPoint(20, 70, 55, 85),
            TrajectoryPoint(24, 75, 60, 90),
        ],
        "qol": [
            TrajectoryPoint(1, 20, 5, 35),
            TrajectoryPoint(2, 30, 15, 45),
            TrajectoryPoint(4, 45, 30, 60),
            TrajectoryPoint(6, 60, 45, 75),
            TrajectoryPoint(8, 70, 55, 85),
            TrajectoryPoint(12, 80, 65, 95),
            TrajectoryPoint(16, 85, 70, 100),
            TrajectoryPoint(20, 90, 75, 100),
            TrajectoryPoint(24, 92, 80, 100),
        ],
    },
}

# ASES Recovery Trajectories (0-100 scale, 100 = best)
ASES_TRAJECTORIES = {
    DiagnosisType.ROTATOR_CUFF_TEAR: {
        "pain_component": [
            TrajectoryPoint(1, 20, 10, 30),
            TrajectoryPoint(2, 25, 15, 35),
            TrajectoryPoint(4, 30, 20, 40),
            TrajectoryPoint(6, 35, 25, 45),
            TrajectoryPoint(8, 40, 30, 50),
            TrajectoryPoint(12, 42, 32, 50),
            TrajectoryPoint(16, 45, 35, 50),
            TrajectoryPoint(20, 47, 37, 50),
            TrajectoryPoint(24, 48, 40, 50),
        ],
        "function_component": [
            TrajectoryPoint(1, 15, 5, 25),
            TrajectoryPoint(2, 20, 10, 30),
            TrajectoryPoint(4, 25, 15, 35),
            TrajectoryPoint(6, 30, 20, 40),
            TrajectoryPoint(8, 35, 25, 45),
            TrajectoryPoint(12, 40, 30, 50),
            TrajectoryPoint(16, 43, 33, 50),
            TrajectoryPoint(20, 45, 35, 50),
            TrajectoryPoint(24, 47, 37, 50),
        ],
        "total_score": [
            TrajectoryPoint(1, 35, 20, 50),
            TrajectoryPoint(2, 45, 30, 60),
            TrajectoryPoint(4, 55, 40, 70),
            TrajectoryPoint(6, 65, 50, 80),
            TrajectoryPoint(8, 75, 60, 90),
            TrajectoryPoint(12, 82, 67, 97),
            TrajectoryPoint(16, 88, 73, 100),
            TrajectoryPoint(20, 92, 80, 100),
            TrajectoryPoint(24, 95, 85, 100),
        ],
    },
    DiagnosisType.LABRAL_TEAR: {
        "pain_component": [
            TrajectoryPoint(1, 25, 15, 35),
            TrajectoryPoint(2, 30, 20, 40),
            TrajectoryPoint(4, 35, 25, 45),
            TrajectoryPoint(6, 40, 30, 50),
            TrajectoryPoint(8, 42, 32, 50),
            TrajectoryPoint(12, 45, 35, 50),
            TrajectoryPoint(16, 47, 37, 50),
            TrajectoryPoint(20, 48, 40, 50),
        ],
        "function_component": [
            TrajectoryPoint(1, 20, 10, 30),
            TrajectoryPoint(2, 25, 15, 35),
            TrajectoryPoint(4, 30, 20, 40),
            TrajectoryPoint(6, 35, 25, 45),
            TrajectoryPoint(8, 40, 30, 50),
            TrajectoryPoint(12, 43, 33, 50),
            TrajectoryPoint(16, 45, 35, 50),
            TrajectoryPoint(20, 47, 37, 50),
        ],
        "total_score": [
            TrajectoryPoint(1, 45, 30, 60),
            TrajectoryPoint(2, 55, 40, 70),
            TrajectoryPoint(4, 65, 50, 80),
            TrajectoryPoint(6, 75, 60, 90),
            TrajectoryPoint(8, 82, 67, 97),
            TrajectoryPoint(12, 88, 73, 100),
            TrajectoryPoint(16, 92, 80, 100),
            TrajectoryPoint(20, 95, 85, 100),
        ],
    },
    DiagnosisType.POST_TOTAL_SHOULDER_REPLACEMENT: {
        "pain_component": [
            TrajectoryPoint(1, 15, 5, 25),
            TrajectoryPoint(2, 20, 10, 30),
            TrajectoryPoint(4, 25, 15, 35),
            TrajectoryPoint(6, 30, 20, 40),
            TrajectoryPoint(8, 35, 25, 45),
            TrajectoryPoint(12, 40, 30, 50),
            TrajectoryPoint(16, 43, 33, 50),
            TrajectoryPoint(20, 45, 35, 50),
            TrajectoryPoint(24, 47, 37, 50),
        ],
        "function_component": [
            TrajectoryPoint(1, 10, 0, 20),
            TrajectoryPoint(2, 15, 5, 25),
            TrajectoryPoint(4, 20, 10, 30),
            TrajectoryPoint(6, 25, 15, 35),
            TrajectoryPoint(8, 30, 20, 40),
            TrajectoryPoint(12, 35, 25, 45),
            TrajectoryPoint(16, 40, 30, 50),
            TrajectoryPoint(20, 43, 33, 50),
            TrajectoryPoint(24, 45, 35, 50),
        ],
        "total_score": [
            TrajectoryPoint(1, 25, 10, 40),
            TrajectoryPoint(2, 35, 20, 50),
            TrajectoryPoint(4, 45, 30, 60),
            TrajectoryPoint(6, 55, 40, 70),
            TrajectoryPoint(8, 65, 50, 80),
            TrajectoryPoint(12, 75, 60, 90),
            TrajectoryPoint(16, 83, 68, 98),
            TrajectoryPoint(20, 88, 73, 100),
            TrajectoryPoint(24, 92, 80, 100),
        ],
    },
}

# Recovery Milestones
RECOVERY_MILESTONES = {
    DiagnosisType.ACL_TEAR: [
        RecoveryMilestone(1, "Basic weight bearing achieved", 45, critical=True, subscale="adl"),
        RecoveryMilestone(2, "Swelling significantly reduced", 50, critical=True, subscale="symptoms"),
        RecoveryMilestone(6, "90° knee flexion achieved", 70, critical=True, subscale="symptoms"),
        RecoveryMilestone(8, "Full weight bearing without pain", 80, critical=True, subscale="pain"),
        RecoveryMilestone(12, "Return to straight-line running", 50, critical=True, subscale="sport"),
        RecoveryMilestone(16, "Sport-specific training initiated", 65, critical=False, subscale="sport"),
        RecoveryMilestone(24, "Return to competitive sport", 80, critical=False, subscale="sport"),
    ],
    DiagnosisType.MENISCUS_TEAR: [
        RecoveryMilestone(1, "Basic mobility restored", 50, critical=True, subscale="adl"),
        RecoveryMilestone(2, "Pain with daily activities resolved", 65, critical=True, subscale="pain"),
        RecoveryMilestone(4, "Full range of motion achieved", 70, critical=True, subscale="symptoms"),
        RecoveryMilestone(8, "Return to impact activities", 75, critical=False, subscale="sport"),
        RecoveryMilestone(12, "Return to sport activities", 85, critical=False, subscale="sport"),
    ],
    DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT: [
        RecoveryMilestone(1, "Basic mobility with assistance", 25, critical=True, subscale="adl"),
        RecoveryMilestone(2, "Independent walking achieved", 35, critical=True, subscale="adl"),
        RecoveryMilestone(6, "Stair climbing ability restored", 65, critical=True, subscale="adl"),
        RecoveryMilestone(8, "Pain significantly reduced", 80, critical=True, subscale="pain"),
        RecoveryMilestone(12, "Return to normal daily activities", 82, critical=False, subscale="adl"),
        RecoveryMilestone(16, "Low-impact recreation possible", 60, critical=False, subscale="sport"),
    ],
    DiagnosisType.ROTATOR_CUFF_TEAR: [
        RecoveryMilestone(1, "Basic arm movement restored", 35, critical=True, subscale="total_score"),
        RecoveryMilestone(2, "Sleep comfort improved", 45, critical=True, subscale="total_score"),
        RecoveryMilestone(6, "Overhead reaching capability", 65, critical=True, subscale="total_score"),
        RecoveryMilestone(8, "Strength returning for daily tasks", 75, critical=True, subscale="total_score"),
        RecoveryMilestone(12, "Return to work activities", 82, critical=False, subscale="total_score"),
        RecoveryMilestone(16, "Return to recreational activities", 88, critical=False, subscale="total_score"),
    ],
    DiagnosisType.LABRAL_TEAR: [
        RecoveryMilestone(1, "Basic shoulder function restored", 45, critical=True, subscale="total_score"),
        RecoveryMilestone(2, "Pain with movement reduced", 55, critical=True, subscale="total_score"),
        RecoveryMilestone(4, "Full range of motion achieved", 65, critical=True, subscale="total_score"),
        RecoveryMilestone(8, "Return to overhead activities", 82, critical=False, subscale="total_score"),
        RecoveryMilestone(12, "Return to sports activities", 88, critical=False, subscale="total_score"),
    ],
    DiagnosisType.POST_TOTAL_SHOULDER_REPLACEMENT: [
        RecoveryMilestone(1, "Basic arm movement with assistance", 25, critical=True, subscale="total_score"),
        RecoveryMilestone(2, "Independent daily activities", 35, critical=True, subscale="total_score"),
        RecoveryMilestone(6, "Overhead reaching capability", 55, critical=True, subscale="total_score"),
        RecoveryMilestone(8, "Strength for daily tasks", 65, critical=True, subscale="total_score"),
        RecoveryMilestone(12, "Return to normal function", 75, critical=False, subscale="total_score"),
        RecoveryMilestone(16, "Return to recreational activities", 83, critical=False, subscale="total_score"),
    ],
}

def get_trajectory_for_diagnosis(diagnosis: DiagnosisType) -> Optional[Dict]:
    """Get recovery trajectory for a specific diagnosis"""
    if diagnosis in KOOS_TRAJECTORIES:
        return KOOS_TRAJECTORIES[diagnosis]
    elif diagnosis in ASES_TRAJECTORIES:
        return ASES_TRAJECTORIES[diagnosis]
    return None

def get_milestones_for_diagnosis(diagnosis: DiagnosisType) -> List[RecoveryMilestone]:
    """Get recovery milestones for a specific diagnosis"""
    return RECOVERY_MILESTONES.get(diagnosis, [])

def get_expected_score_at_week(diagnosis: DiagnosisType, week: int, subscale: str) -> Optional[TrajectoryPoint]:
    """Get expected score for a specific week and subscale"""
    trajectory = get_trajectory_for_diagnosis(diagnosis)
    if not trajectory or subscale not in trajectory:
        return None
    
    # Find the closest trajectory point
    trajectory_points = trajectory[subscale]
    
    # If exact week match
    for point in trajectory_points:
        if point.week == week:
            return point
    
    # Linear interpolation between closest points
    before_point = None
    after_point = None
    
    for point in trajectory_points:
        if point.week < week:
            before_point = point
        elif point.week > week and after_point is None:
            after_point = point
            break
    
    if before_point and after_point:
        # Linear interpolation
        week_diff = after_point.week - before_point.week
        week_offset = week - before_point.week
        ratio = week_offset / week_diff
        
        expected_score = before_point.expected_score + (after_point.expected_score - before_point.expected_score) * ratio
        lower_bound = before_point.lower_bound + (after_point.lower_bound - before_point.lower_bound) * ratio
        upper_bound = before_point.upper_bound + (after_point.upper_bound - before_point.upper_bound) * ratio
        
        return TrajectoryPoint(week, expected_score, lower_bound, upper_bound)
    
    # Use the last available point if beyond trajectory
    if before_point and not after_point:
        return before_point
    
    # Use the first available point if before trajectory
    if after_point and not before_point:
        return after_point
    
    return None

def calculate_weeks_post_surgery(date_of_surgery: datetime, current_date: datetime) -> int:
    """Calculate weeks post-surgery"""
    if not date_of_surgery:
        return 0
    
    days_diff = (current_date - date_of_surgery).days
    return max(0, days_diff // 7)

def is_score_within_corridor(actual_score: float, expected_point: TrajectoryPoint, tolerance: float = 10.0) -> bool:
    """Check if actual score is within the expected recovery corridor"""
    return expected_point.lower_bound <= actual_score <= expected_point.upper_bound

def get_recovery_status_from_trajectory(actual_score: float, expected_point: TrajectoryPoint) -> str:
    """Determine recovery status based on trajectory comparison"""
    if actual_score >= expected_point.upper_bound:
        return "Ahead of Schedule"
    elif actual_score >= expected_point.expected_score:
        return "On Track"
    elif actual_score >= expected_point.lower_bound:
        return "Slightly Behind"
    else:
        return "Behind Schedule"

def get_milestone_status(diagnosis: DiagnosisType, week: int, subscale_scores: Dict[str, float]) -> List[Dict]:
    """Get milestone achievement status"""
    milestones = get_milestones_for_diagnosis(diagnosis)
    milestone_status = []
    
    for milestone in milestones:
        if milestone.week <= week:
            score = subscale_scores.get(milestone.subscale, 0)
            achieved = score >= milestone.expected_score
            milestone_status.append({
                "week": milestone.week,
                "description": milestone.description,
                "expected_score": milestone.expected_score,
                "actual_score": score,
                "achieved": achieved,
                "critical": milestone.critical,
                "subscale": milestone.subscale
            })
    
    return milestone_status