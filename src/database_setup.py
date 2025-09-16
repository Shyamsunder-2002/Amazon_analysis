"""
Database Setup and Integration Module
Handles SQL database creation, data loading, and analytics queries
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import os
from config.settings import DATABASE_PATH, DATABASE_URL

class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.engine = create_engine(DATABASE_URL)
    
    def create_database_schema(self):
        """Create optimized database schema for analytics"""
        print("🗄️ Creating database schema...")
        
        schema_sql = """
        -- Transactions table (main fact table)
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            order_date DATE NOT NULL,
            order_year INTEGER NOT NULL,
            order_month INTEGER NOT NULL,
            order_quarter INTEGER NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            brand TEXT,
            product_name TEXT,
            original_price_inr REAL,
            discount_percent REAL,
            final_amount_inr REAL NOT NULL,
            delivery_charges REAL,
            customer_city TEXT,
            customer_state TEXT,
            age_group TEXT,
            is_prime_member BOOLEAN,
            payment_method TEXT,
            delivery_days INTEGER,
            customer_rating REAL,
            product_rating REAL,
            is_festival_sale BOOLEAN,
            festival_name TEXT,
            return_status TEXT,
            customer_spending_tier TEXT
        );
        
        -- Products catalog table
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            brand TEXT,
            base_price_2015 REAL,
            weight_kg REAL,
            rating REAL,
            is_prime_eligible BOOLEAN,
            launch_year INTEGER,
            model TEXT
        );
        
        -- Customers master table  
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            customer_city TEXT,
            customer_state TEXT,
            age_group TEXT,
            is_prime_member BOOLEAN,
            first_order_date DATE,
            last_order_date DATE,
            total_orders INTEGER,
            total_spent REAL,
            avg_order_value REAL,
            customer_segment TEXT
        );
        
        -- Time dimension table
        CREATE TABLE IF NOT EXISTS time_dimension (
            date_key DATE PRIMARY KEY,
            year INTEGER,
            quarter INTEGER,
            month INTEGER,
            month_name TEXT,
            day INTEGER,
            day_name TEXT,
            week_of_year INTEGER,
            is_weekend BOOLEAN,
            is_festival_season BOOLEAN
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(order_date);
        CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(product_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
        CREATE INDEX IF NOT EXISTS idx_transactions_city ON transactions(customer_city);
        CREATE INDEX IF NOT EXISTS idx_transactions_payment ON transactions(payment_method);
        """
        
        with self.engine.connect() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
                    conn.commit()
        
        print("✅ Database schema created successfully!")
    
    def load_transactions_data(self, df):
        """Load cleaned transactions data into database"""
        print(f"📥 Loading {len(df)} transactions into database...")
        
        # Ensure required columns exist and have correct types
        required_columns = [
            'transaction_id', 'customer_id', 'product_id', 'order_date',
            'category', 'final_amount_inr', 'customer_city'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        # Load data
        df.to_sql('transactions', self.engine, if_exists='replace', index=False, method='multi')
        print("✅ Transactions data loaded successfully!")
        return True
    
    def load_products_data(self, df):
        """Load products catalog data"""
        print(f"📥 Loading {len(df)} products into database...")
        df.to_sql('products', self.engine, if_exists='replace', index=False)
        print("✅ Products data loaded successfully!")
    
    def create_customers_table(self, df):
        """Create and populate customers master table"""
        print("👥 Creating customers master table...")
        
        customers_df = df.groupby('customer_id').agg({
            'customer_city': 'first',
            'customer_state': 'first', 
            'age_group': 'first',
            'is_prime_member': 'first',
            'order_date': ['min', 'max'],
            'transaction_id': 'count',
            'final_amount_inr': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        customers_df.columns = [
            'customer_id', 'customer_city', 'customer_state', 'age_group',
            'is_prime_member', 'first_order_date', 'last_order_date',
            'total_orders', 'total_spent', 'avg_order_value'
        ]
        
        # Add customer segmentation
        def segment_customer(row):
            if row['total_spent'] > 50000 and row['total_orders'] > 10:
                return 'VIP'
            elif row['total_spent'] > 20000 and row['total_orders'] > 5:
                return 'Premium'
            elif row['total_orders'] > 3:
                return 'Regular'
            else:
                return 'New'
        
        customers_df['customer_segment'] = customers_df.apply(segment_customer, axis=1)
        
        customers_df.to_sql('customers', self.engine, if_exists='replace', index=False)
        print(f"✅ Created customers table with {len(customers_df)} customers")
    
    def create_time_dimension(self, start_date='2015-01-01', end_date='2025-12-31'):
        """Create time dimension table"""
        print("📅 Creating time dimension table...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        time_df = pd.DataFrame({
            'date_key': date_range,
            'year': date_range.year,
            'quarter': date_range.quarter,
            'month': date_range.month,
            'month_name': date_range.month_name(),
            'day': date_range.day,
            'day_name': date_range.day_name(),
            'week_of_year': date_range.isocalendar().week,
            'is_weekend': date_range.dayofweek >= 5,
            'is_festival_season': False  # This could be enhanced with actual festival dates
        })
        
        time_df.to_sql('time_dimension', self.engine, if_exists='replace', index=False)
        print(f"✅ Created time dimension with {len(time_df)} dates")
    
    def get_analytics_query(self, query_name):
        """Get pre-defined analytics queries"""
        queries = {
            'revenue_by_year': """
                SELECT 
                    order_year,
                    SUM(final_amount_inr) as total_revenue,
                    COUNT(*) as total_orders,
                    AVG(final_amount_inr) as avg_order_value
                FROM transactions 
                GROUP BY order_year 
                ORDER BY order_year
            """,
            
            'top_categories': """
                SELECT 
                    category,
                    SUM(final_amount_inr) as revenue,
                    COUNT(*) as orders,
                    COUNT(DISTINCT customer_id) as customers
                FROM transactions 
                GROUP BY category 
                ORDER BY revenue DESC
            """,
            
            'customer_segments': """
                SELECT 
                    customer_segment,
                    COUNT(*) as customer_count,
                    AVG(total_spent) as avg_lifetime_value,
                    AVG(total_orders) as avg_order_frequency
                FROM customers 
                GROUP BY customer_segment
            """,
            
            'monthly_trends': """
                SELECT 
                    DATE_FORMAT(order_date, '%Y-%m') as month_year,
                    SUM(final_amount_inr) as revenue,
                    COUNT(*) as orders
                FROM transactions 
                GROUP BY DATE_FORMAT(order_date, '%Y-%m')
                ORDER BY month_year
            """,
            
            'prime_analysis': """
                SELECT 
                    is_prime_member,
                    COUNT(*) as orders,
                    SUM(final_amount_inr) as revenue,
                    AVG(final_amount_inr) as avg_order_value,
                    AVG(customer_rating) as avg_rating
                FROM transactions 
                GROUP BY is_prime_member
            """,
            
            'city_performance': """
                SELECT 
                    customer_city,
                    SUM(final_amount_inr) as revenue,
                    COUNT(DISTINCT customer_id) as customers,
                    AVG(delivery_days) as avg_delivery_days
                FROM transactions 
                GROUP BY customer_city 
                HAVING COUNT(*) >= 100
                ORDER BY revenue DESC
                LIMIT 20
            """
        }
        
        return queries.get(query_name, "")
    
    def execute_query(self, query):
        """Execute SQL query and return DataFrame"""
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            return pd.DataFrame()
    
    def setup_complete_database(self, transactions_df, products_df=None):
        """Complete database setup process"""
        print("🚀 Starting complete database setup...")
        
        # Create schema
        self.create_database_schema()
        
        # Load data
        self.load_transactions_data(transactions_df)
        
        if products_df is not None:
            self.load_products_data(products_df)
        
        # Create derived tables
        self.create_customers_table(transactions_df)
        self.create_time_dimension()
        
        print("🎉 Database setup completed successfully!")
        
        # Verify setup
        with self.engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            print(f"✅ Created tables: {[table[0] for table in tables]}")
        
        return True




