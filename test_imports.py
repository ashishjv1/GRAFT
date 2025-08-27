#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_basic_import():
    """Test basic package import"""
    try:
        import graft
        print("✓ Basic graft package import successful")
        print(f"  Version: {graft.__version__}")
        return True
    except Exception as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_submodule_imports():
    """Test importing submodules"""
    try:
        from graft.models import ResNet18
        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False
        
    try:
        from graft.utils.loader import loader
        print("✓ Utils imports successful") 
    except Exception as e:
        print(f"✗ Utils import failed: {e}")
        return False
        
    return True

def test_main_classes():
    """Test importing main classes"""
    try:
        from graft import ModelTrainer, TrainingConfig
        print("✓ Main classes import successful")
        return True
    except Exception as e:
        print(f"✗ Main classes import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing GRAFT package imports...")
    print("=" * 50)
    
    success = True
    success &= test_basic_import()
    success &= test_main_classes() 
    success &= test_submodule_imports()
    
    print("=" * 50)
    if success:
        print("🎉 All imports successful!")
    else:
        print("❌ Some imports failed")
        exit(1)