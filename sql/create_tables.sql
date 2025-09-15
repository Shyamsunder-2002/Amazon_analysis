-- Amazon India Analytics Database Schema
-- Optimized for analytics and business intelligence

-- =====================================================
-- MAIN TRANSACTIONS TABLE (Fact Table)
-- =====================================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(15) NOT NULL,
    product_id VARCHAR(10) NOT NULL,
    order_date DATE NOT NULL,
    order_year INTEGER NOT NULL,
    order_month INTEGER NOT NULL,
    order_quarter INTEGER NOT NULL,
    order_day_of_week INTEGER,
    
    -- Product Information
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    brand VARCHAR(50),
    product_name VARCHAR(200),
    
    -- Financial Data
    original_price_inr DECIMAL(10,2),
    discount_percent DECIMAL(5,2),
    discount_amount DECIMAL(10,2),
    final_amount_inr DECIMAL(10,2) NOT NULL,
    delivery_charges DECIMAL(8,2),
    
    -- Customer Information
    customer_city VARCHAR(50),
    customer_state VARCHAR(50),
    age_group VARCHAR(20),
    is_prime_member BOOLEAN DEFAULT FALSE,
    
    -- Order Details
    payment_method VARCHAR(30),
    delivery_days INTEGER,
    delivery_date DATE,
    order_status VARCHAR(20) DEFAULT 'Processing',
    
    -- Quality Metrics
    customer_rating DECIMAL(2,1),
    product_rating DECIMAL(2,1),
    
    -- Business Flags
    is_festival_sale BOOLEAN DEFAULT FALSE,
    festival_name VARCHAR(50),
    is_prime_eligible BOOLEAN DEFAULT FALSE,
    return_status VARCHAR(20) DEFAULT 'Not Returned',
    
    -- Operational Data
    warehouse_location VARCHAR(20),
    shipping_partner VARCHAR(30),
    customer_spending_tier VARCHAR(20),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =====================================================
-- PRODUCTS CATALOG TABLE (Dimension Table)
-- =====================================================
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(10) PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    brand VARCHAR(50),
    
    -- Product Attributes
    base_price_2015 DECIMAL(10,2),
    current_price DECIMAL(10,2),
    weight_kg DECIMAL(6,2),
    dimensions_cm VARCHAR(20),
    
    -- Quality Metrics
    avg_rating DECIMAL(2,1),
    total_reviews INTEGER DEFAULT 0,
    
    -- Business Attributes
    is_prime_eligible BOOLEAN DEFAULT FALSE,
    launch_year INTEGER,
    model VARCHAR(100),
    color VARCHAR(30),
    size VARCHAR(20),
    
    -- Inventory
    stock_quantity INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =====================================================
-- CUSTOMERS MASTER TABLE (Dimension Table)
-- =====================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(15) PRIMARY KEY,
    
    -- Demographics
    customer_city VARCHAR(50),
    customer_state VARCHAR(50),
    age_group VARCHAR(20),
    gender VARCHAR(10),
    
    -- Subscription
    is_prime_member BOOLEAN DEFAULT FALSE,
    prime_start_date DATE,
    prime_end_date DATE,
    
    -- Customer Journey
    first_order_date DATE,
    last_order_date DATE,
    total_orders INTEGER DEFAULT 0,
    total_spent DECIMAL(12,2) DEFAULT 0.00,
    avg_order_value DECIMAL(10,2) DEFAULT 0.00,
    
    -- Behavior Metrics
    avg_rating_given DECIMAL(2,1),
    preferred_category VARCHAR(50),
    preferred_payment_method VARCHAR(30),
    
    -- Segmentation
    customer_segment VARCHAR(20),
    customer_tier VARCHAR(20),
    customer_status VARCHAR(20) DEFAULT 'Active',
    
    -- Calculated Metrics
    customer_lifetime_value DECIMAL(12,2) DEFAULT 0.00,
    churn_probability DECIMAL(3,2) DEFAULT 0.00,
    days_since_last_order INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- =====================================================
-- TIME DIMENSION TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS time_dimension (
    date_key DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(12) NOT NULL,
    day INTEGER NOT NULL,
    day_name VARCHAR(12) NOT NULL,
    day_of_year INTEGER,
    week_of_year INTEGER,
    
    -- Business Calendar
    is_weekend BOOLEAN DEFAULT FALSE,
    is_holiday BOOLEAN DEFAULT FALSE,
    is_festival_season BOOLEAN DEFAULT FALSE,
    festival_name VARCHAR(50),
    
    -- Fiscal Calendar
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    fiscal_month INTEGER,
    
    -- Seasonal Indicators
    season VARCHAR(10),
    is_peak_season BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- GEOGRAPHIC DIMENSION TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS geography (
    geo_id INTEGER AUTO_INCREMENT PRIMARY KEY,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    region VARCHAR(30),
    zone VARCHAR(20),
    
    -- Classification
    city_tier VARCHAR(10), -- Metro, Tier1, Tier2, Tier3
    population_category VARCHAR(20),
    
    -- Coordinates (for mapping)
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    
    -- Business Metrics
    market_potential VARCHAR(20),
    delivery_zone VARCHAR(20),
    warehouse_coverage BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_city_state (city, state)
);

-- =====================================================
-- PERFORMANCE INDEXES
-- =====================================================

-- Transactions Table Indexes
CREATE INDEX idx_transactions_date ON transactions(order_date);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_product ON transactions(product_id);
CREATE INDEX idx_transactions_category ON transactions(category);
CREATE INDEX idx_transactions_city ON transactions(customer_city);
CREATE INDEX idx_transactions_payment ON transactions(payment_method);
CREATE INDEX idx_transactions_year_month ON transactions(order_year, order_month);
CREATE INDEX idx_transactions_amount ON transactions(final_amount_inr);
CREATE INDEX idx_transactions_prime ON transactions(is_prime_member);
CREATE INDEX idx_transactions_status ON transactions(order_status);

-- Products Table Indexes  
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_price ON products(current_price);
CREATE INDEX idx_products_rating ON products(avg_rating);
CREATE INDEX idx_products_prime ON products(is_prime_eligible);

-- Customers Table Indexes
CREATE INDEX idx_customers_city ON customers(customer_city);
CREATE INDEX idx_customers_segment ON customers(customer_segment);
CREATE INDEX idx_customers_prime ON customers(is_prime_member);
CREATE INDEX idx_customers_status ON customers(customer_status);
CREATE INDEX idx_customers_tier ON customers(customer_tier);
CREATE INDEX idx_customers_last_order ON customers(last_order_date);

-- Time Dimension Indexes
CREATE INDEX idx_time_year ON time_dimension(year);
CREATE INDEX idx_time_month ON time_dimension(year, month);
CREATE INDEX idx_time_quarter ON time_dimension(year, quarter);
CREATE INDEX idx_time_weekend ON time_dimension(is_weekend);
CREATE INDEX idx_time_festival ON time_dimension(is_festival_season);

-- Geographic Indexes
CREATE INDEX idx_geography_city ON geography(city);
CREATE INDEX idx_geography_state ON geography(state);
CREATE INDEX idx_geography_tier ON geography(city_tier);
