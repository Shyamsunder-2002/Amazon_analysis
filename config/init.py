"""
Configuration Package for Amazon India Analytics Platform
"""

from .settings import *

__version__ = "1.0.0"
__author__ = "Amazon India Analytics Team"
__description__ = "Configuration management for e-commerce analytics platform"

# Configuration validation
def validate_config():
    """Validate configuration settings"""
    import os
    from pathlib import Path
    
    # Check if required directories exist
    required_dirs = [DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, PROCESSED_DATA_DIR]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
    
    # Check database connectivity
    try:
        if DATABASE_PATH.exists():
            print(f"✅ Database found at: {DATABASE_PATH}")
        else:
            print(f"⚠️ Database will be created at: {DATABASE_PATH}")
    except Exception as e:
        print(f"❌ Database configuration error: {e}")
    
    print("📊 Configuration validation completed")

# Auto-validate on import
validate_config()




