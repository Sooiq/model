"""
Setup script for initializing Qlib data
Downloads and prepares data for specified markets
"""

import sys
import qlib
from pathlib import Path
from qlib.contrib.data.handler import check_transform_proc
from qlib.data import simple_dataset


def setup_qlib_us_data():
    """
    Download and setup US market data
    
    This uses Qlib's data downloading tools
    """
    print("Setting up Qlib for US market...")
    
    # Create data directory
    data_path = Path("./data/qlib_data/us")
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Qlib
        provider_uri = str(data_path)
        qlib.init(provider_uri=provider_uri, region="US")
        
        print("Qlib initialized successfully!")
        print(f"Data path: {data_path}")
        print("\nTo download data, run:")
        print("python scripts/download_qlib_data.py --market US --start 2010-01-01")
        
    except Exception as e:
        print(f"Error initializing Qlib: {e}")
        print("\nPlease install Qlib first:")
        print("pip install pyqlib")
        return False
    
    return True


def setup_qlib_korea_data():
    """Setup Korea market data"""
    print("Setting up Qlib for Korea market...")
    print("Note: You may need to use custom data providers for Korean stocks")
    print("Consider using KRX data or yahoo finance for Korean markets")
    

def setup_qlib_indonesia_data():
    """Setup Indonesia market data"""
    print("Setting up Qlib for Indonesia market...")
    print("Note: You may need to use custom data providers for Indonesian stocks")


def setup_qlib_china_data():
    """Setup China market data"""
    print("Setting up Qlib for China market...")
    
    data_path = Path("./data/qlib_data/china")
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        provider_uri = str(data_path)
        qlib.init(provider_uri=provider_uri, region="CN")
        print("Qlib initialized for China market!")
        
    except Exception as e:
        print(f"Error: {e}")


def setup_qlib_uk_data():
    """Setup UK market data"""
    print("Setting up Qlib for UK market...")
    print("Note: You may need to use custom data providers for UK stocks")


def main():
    """Main setup function"""
    print("=" * 60)
    print("SOOIQ Model - Qlib Setup")
    print("=" * 60)
    
    markets = {
        "1": ("US", setup_qlib_us_data),
        "2": ("Korea", setup_qlib_korea_data),
        "3": ("Indonesia", setup_qlib_indonesia_data),
        "4": ("China", setup_qlib_china_data),
        "5": ("UK", setup_qlib_uk_data),
        "6": ("All", None)
    }
    
    print("\nSelect market to setup:")
    for key, (name, _) in markets.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "6":
        # Setup all markets
        for key, (name, func) in markets.items():
            if key != "6" and func:
                print(f"\n{'='*60}")
                func()
    elif choice in markets and markets[choice][1]:
        markets[choice][1]()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download actual market data using Qlib's data tools")
    print("2. Or use the provided data download scripts")
    print("3. Configure data sources in config/data_sources.yaml")


if __name__ == "__main__":
    main()
