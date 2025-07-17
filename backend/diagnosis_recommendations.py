"""
Diagnosis-specific recommendation engine with evidence-based clinical guidelines.
"""

from typing import Dict, List, Optional
from recovery_trajectories import DiagnosisType
from datetime import datetime

class RecommendationEngine:
    """Generate evidence-based recommendations for specific diagnoses"""
    
    # Evidence-based recommendation templates
    KNEE_RECOMMENDATIONS = {
        DiagnosisType.ACL_TEAR: {
            "pain_management": {
                "high": [
                    {
                        "text": "Consider pain management consultation - persistent pain at this stage may indicate complications",
                        "evidence": "MOON cohort studies show early pain intervention improves long-term outcomes",
                        "category": "pain"
                    },
                    {
                        "text": "Ice therapy 15-20 minutes every 2-3 hours to reduce inflammation",
                        "evidence": "Cryotherapy reduces pain and swelling in acute post-operative period",
                        "category": "pain"
                    }
                ],
                "medium": [
                    {
                        "text": "Continue prescribed pain medication regimen as directed",
                        "evidence": "Adequate pain control facilitates rehabilitation participation",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Increase physical therapy frequency - ADL scores suggest functional limitations",
                        "evidence": "Intensive PT in first 12 weeks improves functional outcomes (AOSSM guidelines)",
                        "category": "function"
                    },
                    {
                        "text": "Focus on quadriceps strengthening and range of motion exercises",
                        "evidence": "Early quad activation prevents muscle atrophy and improves outcomes",
                        "category": "function"
                    }
                ],
                "medium": [
                    {
                        "text": "Progress to weight-bearing exercises as tolerated",
                        "evidence": "Progressive loading stimulates healing and prevents stiffness",
                        "category": "function"
                    }
                ]
            },
            "sport_readiness": {
                "high": [
                    {
                        "text": "Initiate sport-specific training program with qualified therapist",
                        "evidence": "Sport-specific training reduces re-injury risk by 51% (2020 AOSSM guidelines)",
                        "category": "activity"
                    },
                    {
                        "text": "Complete return-to-sport testing before clearance",
                        "evidence": "Functional testing reduces ACL re-injury rates",
                        "category": "activity"
                    }
                ]
            },
            "psychological": {
                "high": [
                    {
                        "text": "Consider psychological support for recovery confidence",
                        "evidence": "Fear of re-injury affects 30-40% of ACL patients and impacts outcomes",
                        "category": "general"
                    }
                ]
            }
        },
        DiagnosisType.MENISCUS_TEAR: {
            "pain_management": {
                "high": [
                    {
                        "text": "Evaluate for mechanical symptoms - clicking or locking may indicate incomplete repair",
                        "evidence": "Mechanical symptoms suggest need for arthroscopic evaluation",
                        "category": "pain"
                    }
                ],
                "medium": [
                    {
                        "text": "Anti-inflammatory medications may help reduce pain and swelling",
                        "evidence": "NSAIDs effective for meniscal injury pain management",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Focus on low-impact strengthening exercises",
                        "evidence": "Progressive loading without impact optimizes meniscal healing",
                        "category": "function"
                    }
                ],
                "medium": [
                    {
                        "text": "Gradually increase activity as pain allows",
                        "evidence": "Activity modification promotes healing while maintaining function",
                        "category": "function"
                    }
                ]
            }
        },
        DiagnosisType.POST_TOTAL_KNEE_REPLACEMENT: {
            "pain_management": {
                "high": [
                    {
                        "text": "Persistent pain beyond 8 weeks requires medical evaluation",
                        "evidence": "Chronic pain after TKR may indicate complications or infection",
                        "category": "pain"
                    }
                ],
                "medium": [
                    {
                        "text": "Continue prescribed pain management while increasing activity",
                        "evidence": "Balanced pain control supports rehabilitation progress",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Focus on achieving 90° flexion and full extension",
                        "evidence": "ROM goals critical for functional outcomes after TKR",
                        "category": "function"
                    },
                    {
                        "text": "Emphasize stair climbing and functional mobility training",
                        "evidence": "Functional training improves patient satisfaction and outcomes",
                        "category": "function"
                    }
                ]
            }
        }
    }
    
    SHOULDER_RECOMMENDATIONS = {
        DiagnosisType.ROTATOR_CUFF_TEAR: {
            "pain_management": {
                "high": [
                    {
                        "text": "Consider cortisone injection if conservative measures fail",
                        "evidence": "Corticosteroid injections provide 6-12 weeks pain relief for rotator cuff tears",
                        "category": "pain"
                    },
                    {
                        "text": "Avoid overhead activities that worsen pain",
                        "evidence": "Activity modification prevents further tissue damage",
                        "category": "pain"
                    }
                ],
                "medium": [
                    {
                        "text": "Apply ice after activities to reduce inflammation",
                        "evidence": "Post-activity cryotherapy reduces pain and swelling",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Focus on progressive shoulder elevation exercises",
                        "evidence": "Progressive loading stimulates tendon healing and prevents stiffness",
                        "category": "function"
                    },
                    {
                        "text": "Emphasize scapular stabilization exercises",
                        "evidence": "Scapular stability crucial for rotator cuff function",
                        "category": "function"
                    }
                ],
                "medium": [
                    {
                        "text": "Begin gentle pendulum exercises for mobility",
                        "evidence": "Early passive motion prevents adhesive capsulitis",
                        "category": "function"
                    }
                ]
            }
        },
        DiagnosisType.LABRAL_TEAR: {
            "pain_management": {
                "high": [
                    {
                        "text": "Deep shoulder pain may indicate labral re-tear - consult surgeon",
                        "evidence": "Persistent deep pain suggests incomplete healing",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Avoid extreme range of motion positions initially",
                        "evidence": "Protected ROM prevents re-injury during healing",
                        "category": "function"
                    },
                    {
                        "text": "Progress to sport-specific movements gradually",
                        "evidence": "Gradual return to sport reduces re-injury risk",
                        "category": "function"
                    }
                ]
            }
        },
        DiagnosisType.POST_TOTAL_SHOULDER_REPLACEMENT: {
            "pain_management": {
                "high": [
                    {
                        "text": "Persistent pain beyond 12 weeks warrants evaluation",
                        "evidence": "Chronic pain may indicate complications or loosening",
                        "category": "pain"
                    }
                ]
            },
            "function_deficits": {
                "high": [
                    {
                        "text": "Emphasize forward elevation and external rotation",
                        "evidence": "These motions most important for daily function",
                        "category": "function"
                    },
                    {
                        "text": "Avoid lifting restrictions per surgeon guidelines",
                        "evidence": "Lifting restrictions protect prosthetic integrity",
                        "category": "function"
                    }
                ]
            }
        }
    }
    
    @staticmethod
    def generate_recommendations(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        trajectory_analysis: List[Dict],
        milestone_status: List[Dict],
        concerning_patterns: List[str],
        risk_score: float
    ) -> List[Dict]:
        """Generate personalized recommendations based on current status"""
        
        recommendations = []
        
        # Get diagnosis-specific recommendations
        if diagnosis in RecommendationEngine.KNEE_RECOMMENDATIONS:
            diagnosis_recs = RecommendationEngine.KNEE_RECOMMENDATIONS[diagnosis]
        elif diagnosis in RecommendationEngine.SHOULDER_RECOMMENDATIONS:
            diagnosis_recs = RecommendationEngine.SHOULDER_RECOMMENDATIONS[diagnosis]
        else:
            diagnosis_recs = {}
        
        # Pain-related recommendations
        pain_recommendations = RecommendationEngine._get_pain_recommendations(
            diagnosis, current_scores, diagnosis_recs
        )
        recommendations.extend(pain_recommendations)
        
        # Function-related recommendations
        function_recommendations = RecommendationEngine._get_function_recommendations(
            diagnosis, current_scores, trajectory_analysis, diagnosis_recs
        )
        recommendations.extend(function_recommendations)
        
        # Sport/activity recommendations
        sport_recommendations = RecommendationEngine._get_sport_recommendations(
            diagnosis, weeks_post_surgery, current_scores, diagnosis_recs
        )
        recommendations.extend(sport_recommendations)
        
        # Milestone-based recommendations
        milestone_recommendations = RecommendationEngine._get_milestone_recommendations(
            diagnosis, milestone_status, weeks_post_surgery
        )
        recommendations.extend(milestone_recommendations)
        
        # Risk-based recommendations
        risk_recommendations = RecommendationEngine._get_risk_recommendations(
            diagnosis, risk_score, concerning_patterns
        )
        recommendations.extend(risk_recommendations)
        
        # General recommendations
        general_recommendations = RecommendationEngine._get_general_recommendations(
            diagnosis, weeks_post_surgery, current_scores
        )
        recommendations.extend(general_recommendations)
        
        return recommendations
    
    @staticmethod
    def _get_pain_recommendations(
        diagnosis: DiagnosisType,
        current_scores: Dict[str, float],
        diagnosis_recs: Dict
    ) -> List[Dict]:
        """Generate pain management recommendations"""
        recommendations = []
        
        # Determine pain severity
        pain_score = current_scores.get("pain_score", 0) or current_scores.get("pain_component", 0)
        
        if pain_score < 40:  # Severe pain
            priority = "high"
        elif pain_score < 60:  # Moderate pain
            priority = "medium"
        else:
            priority = "low"
        
        # Get diagnosis-specific pain recommendations
        pain_recs = diagnosis_recs.get("pain_management", {})
        
        if priority in pain_recs:
            for rec in pain_recs[priority]:
                recommendations.append({
                    "text": rec["text"],
                    "priority": priority,
                    "evidence": rec["evidence"],
                    "category": rec["category"]
                })
        
        return recommendations
    
    @staticmethod
    def _get_function_recommendations(
        diagnosis: DiagnosisType,
        current_scores: Dict[str, float],
        trajectory_analysis: List[Dict],
        diagnosis_recs: Dict
    ) -> List[Dict]:
        """Generate function-related recommendations"""
        recommendations = []
        
        # Assess function deficits
        adl_score = current_scores.get("adl_score", 0)
        function_score = current_scores.get("function_component", 0)
        
        # Determine priority based on functional scores
        if adl_score < 50 or function_score < 25:
            priority = "high"
        elif adl_score < 70 or function_score < 35:
            priority = "medium"
        else:
            priority = "low"
        
        # Check for behind-schedule function
        behind_schedule = any(
            t["status"] in ["Behind Schedule", "Slightly Behind"] and 
            t["subscale"] in ["adl_score", "function_component"]
            for t in trajectory_analysis
        )
        
        if behind_schedule:
            priority = "high"
        
        # Get diagnosis-specific function recommendations
        function_recs = diagnosis_recs.get("function_deficits", {})
        
        if priority in function_recs:
            for rec in function_recs[priority]:
                recommendations.append({
                    "text": rec["text"],
                    "priority": priority,
                    "evidence": rec["evidence"],
                    "category": rec["category"]
                })
        
        return recommendations
    
    @staticmethod
    def _get_sport_recommendations(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float],
        diagnosis_recs: Dict
    ) -> List[Dict]:
        """Generate sport/activity recommendations"""
        recommendations = []
        
        sport_score = current_scores.get("sport_score", 0)
        
        # Sport recommendations based on timeline and scores
        if weeks_post_surgery >= 12 and sport_score >= 60:
            sport_recs = diagnosis_recs.get("sport_readiness", {})
            
            if "high" in sport_recs:
                for rec in sport_recs["high"]:
                    recommendations.append({
                        "text": rec["text"],
                        "priority": "high",
                        "evidence": rec["evidence"],
                        "category": rec["category"]
                    })
        
        return recommendations
    
    @staticmethod
    def _get_milestone_recommendations(
        diagnosis: DiagnosisType,
        milestone_status: List[Dict],
        weeks_post_surgery: int
    ) -> List[Dict]:
        """Generate milestone-based recommendations"""
        recommendations = []
        
        # Check for missed critical milestones
        missed_critical = [
            m for m in milestone_status 
            if m["critical"] and not m["achieved"] and m["week"] <= weeks_post_surgery
        ]
        
        for milestone in missed_critical:
            recommendations.append({
                "text": f"Critical milestone missed: {milestone['description']} - consider intervention",
                "priority": "high",
                "evidence": "Missing critical milestones correlates with poor outcomes",
                "category": "general"
            })
        
        return recommendations
    
    @staticmethod
    def _get_risk_recommendations(
        diagnosis: DiagnosisType,
        risk_score: float,
        concerning_patterns: List[str]
    ) -> List[Dict]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.append({
                "text": "High risk score detected - consider comprehensive evaluation",
                "priority": "high",
                "evidence": "Early intervention for high-risk patients improves outcomes",
                "category": "general"
            })
        
        # Pattern-specific recommendations
        for pattern in concerning_patterns:
            if "plateau" in pattern.lower():
                recommendations.append({
                    "text": "Recovery plateau detected - modify treatment approach",
                    "priority": "medium",
                    "evidence": "Treatment modification effective for plateau patterns",
                    "category": "general"
                })
            elif "declining" in pattern.lower():
                recommendations.append({
                    "text": "Declining scores require immediate evaluation",
                    "priority": "high",
                    "evidence": "Score regression indicates need for intervention",
                    "category": "general"
                })
        
        return recommendations
    
    @staticmethod
    def _get_general_recommendations(
        diagnosis: DiagnosisType,
        weeks_post_surgery: int,
        current_scores: Dict[str, float]
    ) -> List[Dict]:
        """Generate general recommendations"""
        recommendations = []
        
        # Timeline-based recommendations
        if weeks_post_surgery <= 4:
            recommendations.append({
                "text": "Continue with early recovery protocols and follow-up appointments",
                "priority": "medium",
                "evidence": "Consistent follow-up improves early recovery outcomes",
                "category": "general"
            })
        elif weeks_post_surgery <= 12:
            recommendations.append({
                "text": "Focus on progressive loading and functional exercises",
                "priority": "medium",
                "evidence": "Mid-recovery phase benefits from progressive activity",
                "category": "general"
            })
        else:
            recommendations.append({
                "text": "Transition to advanced functional and sport-specific training",
                "priority": "medium",
                "evidence": "Late recovery phase should emphasize return to full activity",
                "category": "general"
            })
        
        # Always include PRO assessment recommendation
        recommendations.append({
            "text": "Continue regular PRO assessments for optimal recovery tracking",
            "priority": "low",
            "evidence": "Regular PRO assessment improves recovery monitoring",
            "category": "general"
        })
        
        return recommendations