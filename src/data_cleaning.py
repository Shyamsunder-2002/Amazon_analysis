"""
Comprehensive Data Cleaning Module for Amazon India Analytics
Handles all 10 data cleaning challenges as specified in the project requirements
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.cleaning_reports = {}
    
    def clean_order_dates(self, df, date_column='order_date'):
        """
        Question 1: Clean and standardize order dates
        Handles multiple formats: 'DD/MM/YYYY', 'DD-MM-YY', 'YYYY-MM-DD', invalid entries
        """
        print(f"🗓️ Cleaning {date_column} column...")
        original_count = len(df)
        
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            
            date_str = str(date_str).strip()
            
            # Try different date formats
            formats = [
                '%Y-%m-%d',    # YYYY-MM-DD
                '%d/%m/%Y',    # DD/MM/YYYY
                '%d-%m-%y',    # DD-MM-YY
                '%d/%m/%y',    # DD/MM/YY
                '%d-%m-%Y',    # DD-MM-YYYY
                '%Y/%m/%d',    # YYYY/MM/DD
            ]
            
            for fmt in formats:
                try:
                    parsed_date = pd.to_datetime(date_str, format=fmt)
                    # Check if date is reasonable (between 2015-2025)
                    if 2015 <= parsed_date.year <= 2025:
                        return parsed_date
                except:
                    continue
            
            # If no format works, return NaT
            return pd.NaT
        
        # Apply date cleaning
        df[date_column] = df[date_column].apply(parse_date)
        
        # Remove invalid dates
        valid_dates = df[date_column].notna()
        cleaned_count = valid_dates.sum()
        
        self.cleaning_reports['order_dates'] = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'invalid_count': original_count - cleaned_count,
            'success_rate': f"{(cleaned_count/original_count)*100:.2f}%"
        }
        
        print(f"✅ Date cleaning completed: {cleaned_count}/{original_count} valid dates")
        return df[valid_dates].copy()
    
    def clean_price_columns(self, df, price_columns=['original_price_inr', 'final_amount_inr']):
        """
        Question 2: Clean price columns with mixed formats
        Handles: numeric values, '₹' symbols, comma separators, 'Price on Request'
        """
        print("💰 Cleaning price columns...")
        
        def clean_price(price_str):
            if pd.isna(price_str):
                return np.nan
            
            price_str = str(price_str).strip()
            
            # Handle special cases
            if price_str.lower() in ['price on request', 'contact seller', 'na', 'n/a']:
                return np.nan
            
            # Remove currency symbols and clean
            price_str = re.sub(r'[₹$,\s]', '', price_str)
            
            # Extract numeric value
            numbers = re.findall(r'\d+\.?\d*', price_str)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    return np.nan
            return np.nan
        
        for col in price_columns:
            if col in df.columns:
                original_valid = df[col].notna().sum()
                df[col] = df[col].apply(clean_price)
                cleaned_valid = df[col].notna().sum()
                
                self.cleaning_reports[f'price_{col}'] = {
                    'original_valid': original_valid,
                    'cleaned_valid': cleaned_valid,
                    'improvement': cleaned_valid - original_valid
                }
                print(f"✅ {col}: {cleaned_valid} valid prices")
        
        return df
    
    def clean_ratings(self, df, rating_columns=['customer_rating', 'product_rating']):
        """
        Question 3: Standardize ratings to 1.0-5.0 scale
        Handles: '5.0', '4 stars', '3/5', '2.5/5.0', missing values
        """
        print("⭐ Cleaning rating columns...")
        
        def standardize_rating(rating_str):
            if pd.isna(rating_str):
                return np.nan
            
            rating_str = str(rating_str).strip().lower()
            
            # Handle different formats
            if 'stars' in rating_str or 'star' in rating_str:
                # Extract number before 'stars'
                numbers = re.findall(r'\d+\.?\d*', rating_str)
                if numbers:
                    return min(float(numbers[0]), 5.0)
            
            elif '/' in rating_str:
                # Handle X/Y format
                parts = rating_str.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        # Convert to 5-point scale
                        return min((numerator / denominator) * 5, 5.0)
                    except:
                        return np.nan
            
            else:
                # Direct numeric value
                try:
                    value = float(rating_str)
                    return min(max(value, 1.0), 5.0)  # Clamp between 1-5
                except:
                    return np.nan
            
            return np.nan
        
        for col in rating_columns:
            if col in df.columns:
                df[col] = df[col].apply(standardize_rating)
                # Fill missing ratings with category median
                median_rating = df[col].median()
                df[col] = df[col].fillna(median_rating)
                print(f"✅ {col} standardized to 1.0-5.0 scale")
        
        return df
    
    def standardize_cities(self, df, city_column='customer_city'):
        """
        Question 4: Standardize city names
        Handles: 'Bangalore/Bengaluru', 'Mumbai/Bombay', spelling errors, case variations
        """
        print("🏙️ Standardizing city names...")
        
        # City name mappings
        city_mappings = {
            'bengaluru': 'Bangalore',
            'bangalore': 'Bangalore',
            'bombay': 'Mumbai',
            'mumbai': 'Mumbai',
            'new delhi': 'Delhi',
            'delhi': 'Delhi',
            'calcutta': 'Kolkata',
            'kolkata': 'Kolkata',
            'madras': 'Chennai',
            'chennai': 'Chennai',
            'mysore': 'Mysuru',
            'mysuru': 'Mysuru',
            'cochin': 'Kochi',
            'kochi': 'Kochi',
        }
        
        def standardize_city(city_name):
            if pd.isna(city_name):
                return 'Unknown'
            
            city_name = str(city_name).strip().lower()
            
            # Handle slash separated names
            if '/' in city_name:
                city_name = city_name.split('/')[0]
            
            # Apply mappings
            standardized = city_mappings.get(city_name, city_name.title())
            return standardized
        
        df[city_column] = df[city_column].apply(standardize_city)
        unique_cities = df[city_column].nunique()
        
        self.cleaning_reports['cities'] = {
            'unique_cities': unique_cities,
            'standardized': True
        }
        
        print(f"✅ Standardized {unique_cities} unique cities")
        return df
    
    def clean_boolean_columns(self, df, boolean_columns=['is_prime_member', 'is_prime_eligible', 'is_festival_sale']):
        """
        Question 5: Standardize boolean columns
        Handles: True/False, Yes/No, 1/0, Y/N, missing entries
        """
        print("✅ Cleaning boolean columns...")
        
        def standardize_boolean(value):
            if pd.isna(value):
                return False  # Default to False for missing values
            
            value_str = str(value).strip().lower()
            
            if value_str in ['true', '1', 'yes', 'y', '1.0']:
                return True
            elif value_str in ['false', '0', 'no', 'n', '0.0']:
                return False
            else:
                return False  # Default for unclear values
        
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].apply(standardize_boolean)
                true_count = df[col].sum()
                print(f"✅ {col}: {true_count} True values")
        
        return df
    
    def standardize_categories(self, df, category_columns=['category', 'subcategory']):
        """
        Question 6: Standardize product categories
        Handles: 'Electronics/Electronic/ELECTRONICS/Electronics & Accessories'
        """
        print("📦 Standardizing product categories...")
        
        category_mappings = {
            'electronics': 'Electronics',
            'electronic': 'Electronics',
            'electronics & accessories': 'Electronics',
            'clothing': 'Clothing & Accessories',
            'clothes': 'Clothing & Accessories',
            'apparel': 'Clothing & Accessories',
            'home': 'Home & Kitchen',
            'home & kitchen': 'Home & Kitchen',
            'kitchen': 'Home & Kitchen',
            'sports': 'Sports & Outdoors',
            'outdoor': 'Sports & Outdoors',
            'books': 'Books',
            'book': 'Books',
            'beauty': 'Beauty & Personal Care',
            'personal care': 'Beauty & Personal Care',
            'toys': 'Toys & Games',
            'games': 'Toys & Games',
            'automotive': 'Automotive',
            'auto': 'Automotive'
        }
        
        def standardize_category(category):
            if pd.isna(category):
                return 'Other'
            
            category = str(category).strip().lower()
            
            # Handle slash separated categories
            if '/' in category:
                category = category.split('/')[0]
            
            return category_mappings.get(category, category.title())
        
        for col in category_columns:
            if col in df.columns:
                df[col] = df[col].apply(standardize_category)
                unique_categories = df[col].nunique()
                print(f"✅ {col}: {unique_categories} unique categories")
        
        return df
    
    def clean_delivery_days(self, df, delivery_column='delivery_days'):
        """
        Question 7: Clean delivery days column
        Handles: negative values, 'Same Day', '1-2 days', unrealistic values
        """
        print("🚚 Cleaning delivery days...")
        
        def clean_delivery(delivery_str):
            if pd.isna(delivery_str):
                return 3  # Default delivery time
            
            delivery_str = str(delivery_str).strip().lower()
            
            if delivery_str in ['same day', 'same-day', '0']:
                return 0
            elif 'next day' in delivery_str or delivery_str == '1':
                return 1
            elif '-' in delivery_str:
                # Handle range like '1-2 days'
                numbers = re.findall(r'\d+', delivery_str)
                if len(numbers) >= 2:
                    return (int(numbers[0]) + int(numbers[1])) / 2
                elif len(numbers) == 1:
                    return int(numbers[0])
            else:
                # Extract numeric value
                numbers = re.findall(r'\d+', delivery_str)
                if numbers:
                    days = int(numbers[0])
                    # Cap unrealistic values
                    return min(max(days, 0), 14)  # Max 14 days
            
            return 3  # Default
        
        df[delivery_column] = df[delivery_column].apply(clean_delivery)
        avg_delivery = df[delivery_column].mean()
        
        self.cleaning_reports['delivery_days'] = {
            'average_delivery_days': f"{avg_delivery:.2f}",
            'max_delivery_days': df[delivery_column].max(),
            'min_delivery_days': df[delivery_column].min()
        }
        
        print(f"✅ Delivery days cleaned. Average: {avg_delivery:.2f} days")
        return df
    
    def handle_duplicates(self, df, key_columns=['customer_id', 'product_id', 'order_date', 'final_amount_inr']):
        """
        Question 8: Handle duplicate transactions
        Distinguishes between genuine bulk orders and data errors
        """
        print("🔍 Handling duplicate transactions...")
        
        original_count = len(df)
        
        # Identify potential duplicates
        duplicates_mask = df.duplicated(subset=key_columns, keep=False)
        duplicate_groups = df[duplicates_mask].groupby(key_columns).size()
        
        # Strategy: Keep genuine bulk orders, remove data errors
        def is_genuine_bulk(group):
            if len(group) <= 3:  # Up to 3 same items might be genuine
                return True
            # Large quantities might be genuine for certain categories
            if group['category'].iloc[0] in ['Books', 'Beauty & Personal Care']:
                return len(group) <= 5
            return False
        
        # Remove problematic duplicates
        to_remove = []
        for name, group in df.groupby(key_columns):
            if len(group) > 1 and not is_genuine_bulk(group):
                # Keep only the first occurrence
                to_remove.extend(group.index[1:].tolist())
        
        df_cleaned = df.drop(to_remove)
        removed_count = len(to_remove)
        
        self.cleaning_reports['duplicates'] = {
            'original_count': original_count,
            'removed_duplicates': removed_count,
            'final_count': len(df_cleaned),
            'duplicate_rate': f"{(removed_count/original_count)*100:.2f}%"
        }
        
        print(f"✅ Removed {removed_count} duplicate records")
        return df_cleaned
    
    def fix_price_outliers(self, df, price_columns=['original_price_inr', 'final_amount_inr']):
        """
        Question 9: Fix price outliers using statistical methods
        Identifies and corrects decimal point errors and unrealistic prices
        """
        print("📊 Fixing price outliers...")
        
        for col in price_columns:
            if col not in df.columns:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Fix decimal point errors (prices 100x too high)
            high_outliers = df[col] > upper_bound * 10
            df.loc[high_outliers, col] = df.loc[high_outliers, col] / 100
            
            # Cap remaining extreme values
            df.loc[df[col] < 10, col] = df[col].median()  # Minimum reasonable price
            df.loc[df[col] > 1000000, col] = df[col].median()  # Maximum reasonable price
            
            outliers_fixed = high_outliers.sum()
            print(f"✅ {col}: Fixed {outliers_fixed} outliers")
        
        return df
    
    def standardize_payment_methods(self, df, payment_column='payment_method'):
        """
        Question 10: Standardize payment method categories
        Handles: 'UPI/PhonePe/GooglePay', 'Credit Card/CREDIT_CARD/CC'
        """
        print("💳 Standardizing payment methods...")
        
        payment_mappings = {
            'phonepe': 'UPI',
            'googlepay': 'UPI',
            'paytm': 'UPI',
            'upi': 'UPI',
            'credit card': 'Credit Card',
            'credit_card': 'Credit Card',
            'cc': 'Credit Card',
            'debit card': 'Debit Card',
            'debit_card': 'Debit Card',
            'dc': 'Debit Card',
            'net banking': 'Net Banking',
            'netbanking': 'Net Banking',
            'cash on delivery': 'Cash on Delivery',
            'cod': 'Cash on Delivery',
            'c.o.d': 'Cash on Delivery',
            'wallet': 'Wallet'
        }
        
        def standardize_payment(payment):
            if pd.isna(payment):
                return 'Other'
            
            payment = str(payment).strip().lower()
            
            # Handle slash separated methods
            if '/' in payment:
                payment = payment.split('/')[0]
            
            return payment_mappings.get(payment, payment.title())
        
        df[payment_column] = df[payment_column].apply(standardize_payment)
        unique_methods = df[payment_column].value_counts().to_frame()
        
        self.cleaning_reports['payment_methods'] = {
            'unique_methods': len(unique_methods),
            'method_distribution': unique_methods.to_dict()
        }
        
        print(f"✅ Standardized to {len(unique_methods)} payment methods")
        return df
    
    def generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print("\n" + "="*50)
        print("🧹 DATA CLEANING SUMMARY REPORT")
        print("="*50)
        
        for operation, stats in self.cleaning_reports.items():
            print(f"\n{operation.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return self.cleaning_reports
    
    def clean_complete_dataset(self, df):
        """Execute all cleaning operations in sequence"""
        print("🚀 Starting comprehensive data cleaning pipeline...")
        print(f"📊 Initial dataset shape: {df.shape}")
        
        # Execute all cleaning steps
        df = self.clean_order_dates(df)
        df = self.clean_price_columns(df)
        df = self.clean_ratings(df)
        df = self.standardize_cities(df)
        df = self.clean_boolean_columns(df)
        df = self.standardize_categories(df)
        df = self.clean_delivery_days(df)
        df = self.handle_duplicates(df)
        df = self.fix_price_outliers(df)
        df = self.standardize_payment_methods(df)
        
        print(f"✅ Final dataset shape: {df.shape}")
        print("🎉 Data cleaning pipeline completed successfully!")
        
        # Generate report
        self.generate_cleaning_report()
        
        return df

# Example usage function for testing
def test_data_cleaning():
    """Test the data cleaning pipeline"""
    # This would be called with actual data
    cleaner = DataCleaner()
    # df_cleaned = cleaner.clean_complete_dataset(raw_df)
    return cleaner




