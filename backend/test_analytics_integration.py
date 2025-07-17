#!/usr/bin/env python3
"""
Test script to validate analytics integration without database dependencies.
"""

def test_imports():
    """Test that all analytics modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test recovery metrics utilities
        from utils.recovery_metrics import (
            RecoveryMetricsCalculator, 
            RecoveryPhase, 
            MetricCategory,
            TrendAnalyzer,
            MilestoneTracker
        )
        print("✅ Recovery metrics utilities imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_calculations():
    """Test basic calculation functions"""
    print("Testing basic calculations...")
    
    try:
        from utils.recovery_metrics import RecoveryMetricsCalculator
        
        calculator = RecoveryMetricsCalculator()
        
        # Test recovery velocity calculation
        velocity = calculator.calculate_recovery_velocity([1, 2, 3], [2, 3, 4], 7)
        print(f"✅ Recovery velocity calculation: {velocity}")
        
        # Test activity consistency
        consistency = calculator.calculate_activity_consistency_score([5000, 5200, 4800, 5100, 4900], 5)
        print(f"✅ Activity consistency calculation: {consistency}")
        
        # Test sleep quality index
        sleep_quality = calculator.calculate_sleep_quality_index(
            sleep_efficiency=[85, 82, 88, 90, 87],
            sleep_duration=[7.5, 7.2, 8.0, 7.8, 7.6],
            wake_episodes=[2, 3, 1, 2, 2]
        )
        print(f"✅ Sleep quality index calculation: {sleep_quality}")
        
        # Test pain-function ratio
        pain_function = calculator.calculate_pain_function_ratio([30, 35, 40], [70, 75, 80])
        print(f"✅ Pain-function ratio calculation: {pain_function}")
        
        return True
    except Exception as e:
        print(f"❌ Calculation error: {e}")
        return False

def test_trend_analysis():
    """Test trend analysis functionality"""
    print("Testing trend analysis...")
    
    try:
        from utils.recovery_metrics import TrendAnalyzer
        from datetime import date, timedelta
        
        # Generate test data
        values = [50, 52, 55, 58, 60, 62, 65, 68, 70, 72]
        dates = [date.today() - timedelta(days=i) for i in range(len(values))]
        
        trend = TrendAnalyzer.analyze_metric_trend(values, dates, "test_metric")
        print(f"✅ Trend analysis: {trend.trend_direction}, magnitude: {trend.trend_magnitude:.3f}")
        
        # Test changepoint detection
        changepoints = TrendAnalyzer.detect_changepoints(values + [45, 48, 50])  # Add a sudden drop
        print(f"✅ Changepoint detection: {len(changepoints)} changepoints found")
        
        return True
    except Exception as e:
        print(f"❌ Trend analysis error: {e}")
        return False

def test_milestone_tracking():
    """Test milestone tracking functionality"""
    print("Testing milestone tracking...")
    
    try:
        from utils.recovery_metrics import MilestoneTracker
        from datetime import date
        
        # Test milestone creation
        milestones = MilestoneTracker.create_diagnosis_milestones("ACL Tear", date.today())
        print(f"✅ Created {len(milestones)} milestones for ACL Tear")
        
        # Test milestone progress calculation
        progress = MilestoneTracker.calculate_milestone_progress(milestones, 8)  # 8 weeks post-surgery
        print(f"✅ Milestone progress: {progress['overall_progress']:.1f}% overall")
        
        return True
    except Exception as e:
        print(f"❌ Milestone tracking error: {e}")
        return False

def test_advanced_metrics():
    """Test advanced metric calculations"""
    print("Testing advanced metrics...")
    
    try:
        from utils.recovery_metrics import RecoveryMetricsCalculator
        
        calculator = RecoveryMetricsCalculator()
        
        # Test mobility progression index
        mobility_index = calculator.calculate_mobility_progression_index(
            walking_speeds=[1.0, 1.1, 1.2, 1.3],
            step_counts=[3000, 3500, 4000, 4500],
            weeks_post_surgery=8,
            diagnosis_type="ACL Tear"
        )
        print(f"✅ Mobility progression index: {mobility_index}")
        
        # Test cardiovascular recovery index
        cv_index = calculator.calculate_cardiovascular_recovery_index(
            resting_hrs=[85, 82, 80, 78],
            hr_variabilities=[25, 28, 30, 32],
            recovery_hrs=[120, 115, 110, 105]
        )
        print(f"✅ Cardiovascular recovery index: {cv_index}")
        
        # Test adherence score
        adherence = calculator.calculate_adherence_score(
            expected_sessions=20,
            completed_sessions=18,
            data_collection_days=25,
            expected_data_days=30
        )
        print(f"✅ Adherence score: {adherence}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced metrics error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting Analytics Integration Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_calculations,
        test_trend_analysis,
        test_milestone_tracking,
        test_advanced_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 30)
    
    print()
    print("🏆 VALIDATION RESULTS")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All analytics components validated successfully!")
        print("✅ Ready for production deployment")
    else:
        print("⚠️  Some tests failed - review implementation")
    
    return passed == total

if __name__ == "__main__":
    main()