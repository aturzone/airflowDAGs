#!/usr/bin/env python3
"""
Validation script to check dashboard code before deployment
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_manager():
    """Test config manager can load config"""
    print("Testing ConfigManager...")
    try:
        from dashboard.utils.config_manager import ConfigManager

        # Test loading config
        config_path = project_root / 'config' / 'config.yaml'
        cm = ConfigManager(str(config_path))

        # Test basic operations
        assert cm.config is not None, "Config not loaded"
        assert 'databases' in cm.config, "Missing databases section"
        assert 'airflow' in cm.config, "Missing airflow section"
        assert 'models' in cm.config, "Missing models section"
        assert 'ensemble' in cm.config, "Missing ensemble section"

        # Test getters
        ch_config = cm.get_clickhouse_config()
        assert ch_config is not None, "ClickHouse config not found"

        # Test validation
        valid, errors = cm.validate_config()
        if not valid:
            print(f"  Warning: Config validation errors: {errors}")

        print("  ✅ ConfigManager OK")
        return True
    except Exception as e:
        print(f"  ❌ ConfigManager failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test all manager imports"""
    print("Testing module imports...")

    modules = [
        'dashboard.utils.config_manager',
        'dashboard.utils.database_manager',
        'dashboard.utils.airflow_manager',
        'dashboard.utils.docker_manager',
        'dashboard.utils.model_manager',
    ]

    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            all_ok = False

    return all_ok

def test_yaml_syntax():
    """Test YAML config file syntax"""
    print("Testing config YAML syntax...")
    try:
        import yaml
        config_path = project_root / 'config' / 'config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config is not None, "Empty config"
        print("  ✅ YAML syntax OK")
        return True
    except Exception as e:
        print(f"  ❌ YAML syntax error: {e}")
        return False

def test_dashboard_pages():
    """Test dashboard page files exist and have basic syntax"""
    print("Testing dashboard pages...")

    pages_dir = project_root / 'dashboard' / 'pages'
    expected_pages = [
        '1_Dashboard.py',
        '2_Models.py',
        '3_DAGs.py',
        '4_Data.py',
        '5_Anomalies.py',
        '6_Configuration.py',
        '7_Services.py',
        '8_Analytics.py',
    ]

    all_ok = True
    for page in expected_pages:
        page_path = pages_dir / page
        if page_path.exists():
            # Try to compile the file
            try:
                with open(page_path, 'r') as f:
                    compile(f.read(), str(page_path), 'exec')
                print(f"  ✅ {page}")
            except SyntaxError as e:
                print(f"  ❌ {page}: Syntax error - {e}")
                all_ok = False
        else:
            print(f"  ❌ {page}: Missing")
            all_ok = False

    return all_ok

def test_main_app():
    """Test main app.py file"""
    print("Testing main app.py...")
    try:
        app_path = project_root / 'dashboard' / 'app.py'
        with open(app_path, 'r') as f:
            compile(f.read(), str(app_path), 'exec')
        print("  ✅ app.py syntax OK")
        return True
    except Exception as e:
        print(f"  ❌ app.py error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("AnomalyGuard Dashboard Validation")
    print("=" * 60)
    print()

    tests = [
        ("YAML Config", test_yaml_syntax),
        ("Module Imports", test_imports),
        ("Config Manager", test_config_manager),
        ("Dashboard Pages", test_dashboard_pages),
        ("Main App", test_main_app),
    ]

    results = {}
    for name, test_func in tests:
        print()
        results[name] = test_func()

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print()
        print("🎉 All validations passed! Ready to deploy.")
        return 0
    else:
        print()
        print("⚠️  Some validations failed. Please fix issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
