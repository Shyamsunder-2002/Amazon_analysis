"""
Configuration settings for Amazon India Analytics Platform
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database settings
DATABASE_URL = f"sqlite:///{PROCESSED_DATA_DIR}/analytics_data.db"
DATABASE_PATH = PROCESSED_DATA_DIR / "analytics_data.db"

# Streamlit settings
PAGE_TITLE = "🛒 Amazon India: A Decade of Sales Analytics"
PAGE_ICON = "📈"
LAYOUT = "wide"

# Data files
MAIN_DATA_FILE = "amazon_india_complete_2015_2025.csv"
PRODUCTS_FILE = "amazon_india_products_catalog.csv"

# Visualization settings
PLOTLY_THEME = "plotly_white"
COLOR_PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]

# Business constants
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
    "Andaman and Nicobar Islands", "Dadra and Nagar Haveli", "Daman and Diu", "Lakshadweep"
]

METRO_CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune"]
TIER_1_CITIES = ["Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna"]

PRODUCT_CATEGORIES = [
    "Electronics", "Clothing & Accessories", "Home & Kitchen", "Sports & Outdoors",
    "Books", "Beauty & Personal Care", "Toys & Games", "Automotive"
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery", "Wallet"]

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, CLEANED_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)




