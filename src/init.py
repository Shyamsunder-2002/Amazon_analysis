"""
Source Package for Amazon India Analytics Platform
Core modules for data processing, analysis, and visualization
"""

from .data_cleaning import DataCleaner
from .eda_analysis import EDAAnalyzer
from .database_setup import DatabaseManager
from .visualization_utils import *
from .dashboard_components import *

__version__ = "1.0.0"
__author__ = "Amazon India Analytics Team"
__description__ = "Core analytics and data processing modules"

# Package metadata
MODULES = {
    'data_cleaning': 'Advanced data preprocessing and quality enhancement',
    'eda_analysis': 'Exploratory data analysis and statistical insights',
    'database_setup': 'Database management and SQL operations',
    'visualization_utils': 'Plotting utilities and chart styling',
    'dashboard_components': 'Reusable dashboard elements'
}

def get_module_info():
    """Get information about available modules"""
    print("📦 Amazon India Analytics - Source Modules")
    print("=" * 50)
    
    for module, description in MODULES.items():
        print(f"🔧 {module}: {description}")
    
    print(f"\n📊 Package Version: {__version__}")
    print(f"👥 Author: {__author__}")

def run_diagnostics():
    """Run diagnostics on all modules"""
    import importlib
    
    print("🔍 Running Module Diagnostics...")
    print("-" * 40)
    
    for module_name in MODULES.keys():
        try:
            module = importlib.import_module(f'.{module_name}', package='src')
            print(f"✅ {module_name}: OK")
        except ImportError as e:
            print(f"❌ {module_name}: Import Error - {e}")
        except Exception as e:
            print(f"⚠️ {module_name}: Warning - {e}")
    
    print("\n🎉 Diagnostics completed!")

# Auto-run diagnostics on import (optional)
# run_diagnostics()




