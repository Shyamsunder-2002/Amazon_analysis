-- Analytics Queries for Business Intelligence
-- Amazon India Analytics Platform

-- =====================================================
-- EXECUTIVE DASHBOARD QUERIES
-- =====================================================

-- Key Performance Indicators
CREATE OR REPLACE VIEW executive_kpis AS
SELECT 
    COUNT(DISTINCT t.customer_id) as total_customers,
    COUNT(*) as total_orders,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_order_value,
    SUM(CASE WHEN t.is_prime_member = 1 THEN t.final_amount_inr ELSE 0 END) / SUM(t.final_amount_inr) * 100 as prime_revenue_share,
    AVG(t.customer_rating) as avg_customer_satisfaction,
    AVG(t.delivery_days) as avg_delivery_days,
    SUM(CASE WHEN t.order_status = 'Delivered' THEN 1 ELSE 0 END) / COUNT(*) * 100 as order_completion_rate
FROM transactions t
WHERE t.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR);

-- Revenue Trends (Monthly)
CREATE OR REPLACE VIEW monthly_revenue_trends AS
SELECT 
    t.order_year,
    t.order_month,
    CONCAT(t.order_year, '-', LPAD(t.order_month, 2, '0')) as year_month,
    COUNT(*) as total_orders,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_order_value,
    SUM(CASE WHEN t.is_prime_member = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100 as prime_penetration
FROM transactions t
GROUP BY t.order_year, t.order_month
ORDER BY t.order_year, t.order_month;

-- =====================================================
-- REVENUE ANALYTICS QUERIES
-- =====================================================

-- Category Performance
CREATE OR REPLACE VIEW category_performance AS
SELECT 
    t.category,
    COUNT(*) as total_orders,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    COUNT(DISTINCT t.product_id) as unique_products,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_order_value,
    AVG(t.product_rating) as avg_product_rating,
    AVG(t.customer_rating) as avg_customer_rating,
    SUM(CASE WHEN t.return_status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*) * 100 as return_rate,
    AVG(t.discount_percent) as avg_discount_percent
FROM transactions t
GROUP BY t.category
ORDER BY total_revenue DESC;

-- Geographic Performance
CREATE OR REPLACE VIEW geographic_performance AS
SELECT 
    t.customer_state,
    t.customer_city,
    COUNT(*) as total_orders,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_order_value,
    AVG(t.delivery_days) as avg_delivery_days,
    AVG(t.customer_rating) as avg_customer_satisfaction,
    SUM(CASE WHEN t.is_prime_member = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100 as prime_penetration
FROM transactions t
GROUP BY t.customer_state, t.customer_city
HAVING COUNT(*) >= 100
ORDER BY total_revenue DESC;

-- =====================================================
-- CUSTOMER ANALYTICS QUERIES  
-- =====================================================

-- Customer Segmentation (RFM Analysis)
CREATE OR REPLACE VIEW customer_rfm_analysis AS
WITH customer_rfm AS (
    SELECT 
        c.customer_id,
        DATEDIFF(CURDATE(), c.last_order_date) as recency,
        c.total_orders as frequency,
        c.total_spent as monetary,
        c.customer_city,
        c.customer_state,
        c.age_group,
        c.is_prime_member
    FROM customers c
    WHERE c.customer_status = 'Active'
),
rfm_scores AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency DESC) as r_score,
        NTILE(5) OVER (ORDER BY frequency) as f_score,
        NTILE(5) OVER (ORDER BY monetary) as m_score
    FROM customer_rfm
)
SELECT 
    *,
    CASE 
        WHEN f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 THEN 'New Customers'
        WHEN r_score <= 2 THEN 'At Risk'
        WHEN f_score <= 2 THEN 'Price Sensitive'
        ELSE 'Potential Loyalists'
    END as customer_segment
FROM rfm_scores;

-- Customer Lifetime Value
CREATE OR REPLACE VIEW customer_lifetime_value AS
SELECT 
    c.customer_id,
    c.customer_city,
    c.age_group,
    c.is_prime_member,
    c.total_orders,
    c.total_spent,
    c.avg_order_value,
    DATEDIFF(c.last_order_date, c.first_order_date) as customer_lifespan_days,
    c.total_spent / GREATEST(DATEDIFF(c.last_order_date, c.first_order_date), 1) as revenue_per_day,
    c.total_spent + (c.avg_order_value * c.total_orders * 0.2) as predicted_clv,
    CASE 
        WHEN c.total_spent > 100000 THEN 'Premium'
        WHEN c.total_spent > 50000 THEN 'High Value'
        WHEN c.total_spent > 10000 THEN 'Medium Value'
        ELSE 'Low Value'
    END as value_segment
FROM customers c
WHERE c.total_orders > 0;

-- =====================================================
-- PRODUCT ANALYTICS QUERIES
-- =====================================================

-- Product Performance Analysis
CREATE OR REPLACE VIEW product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.brand,
    COUNT(t.transaction_id) as total_sales,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_selling_price,
    AVG(t.product_rating) as avg_product_rating,
    AVG(t.customer_rating) as avg_customer_rating,
    SUM(CASE WHEN t.return_status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*) * 100 as return_rate,
    AVG(t.discount_percent) as avg_discount_percent,
    COUNT(DISTINCT t.customer_id) as unique_customers
FROM products p
LEFT JOIN transactions t ON p.product_id = t.product_id
GROUP BY p.product_id, p.product_name, p.category, p.brand
HAVING COUNT(t.transaction_id) > 0
ORDER BY total_revenue DESC;

-- Brand Performance
CREATE OR REPLACE VIEW brand_performance AS
SELECT 
    t.brand,
    t.category,
    COUNT(*) as total_sales,
    COUNT(DISTINCT t.product_id) as unique_products,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_selling_price,
    AVG(t.product_rating) as avg_product_rating,
    AVG(t.customer_rating) as avg_customer_rating,
    SUM(CASE WHEN t.return_status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*) * 100 as return_rate
FROM transactions t
GROUP BY t.brand, t.category
ORDER BY total_revenue DESC;

-- =====================================================
-- OPERATIONS ANALYTICS QUERIES
-- =====================================================

-- Delivery Performance
CREATE OR REPLACE VIEW delivery_performance AS
SELECT 
    t.customer_city,
    t.shipping_partner,
    COUNT(*) as total_deliveries,
    AVG(t.delivery_days) as avg_delivery_days,
    SUM(CASE WHEN t.delivery_days <= 3 THEN 1 ELSE 0 END) / COUNT(*) * 100 as on_time_delivery_rate,
    AVG(t.delivery_charges) as avg_delivery_charges,
    AVG(t.customer_rating) as avg_customer_satisfaction,
    SUM(CASE WHEN t.order_status = 'Delivered' THEN 1 ELSE 0 END) / COUNT(*) * 100 as delivery_success_rate
FROM transactions t
WHERE t.delivery_days IS NOT NULL
GROUP BY t.customer_city, t.shipping_partner
HAVING COUNT(*) >= 50
ORDER BY avg_customer_satisfaction DESC;

-- Payment Method Analysis
CREATE OR REPLACE VIEW payment_method_analysis AS
SELECT 
    t.payment_method,
    t.order_year,
    COUNT(*) as total_transactions,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_transaction_value,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY t.order_year) as market_share_percent
FROM transactions t
GROUP BY t.payment_method, t.order_year
ORDER BY t.order_year, total_revenue DESC;

-- =====================================================
-- ADVANCED ANALYTICS QUERIES
-- =====================================================

-- Market Basket Analysis (Simplified)
CREATE OR REPLACE VIEW market_basket_analysis AS
WITH category_pairs AS (
    SELECT 
        t1.customer_id,
        t1.order_date,
        t1.category as category_a,
        t2.category as category_b
    FROM transactions t1
    JOIN transactions t2 ON t1.customer_id = t2.customer_id 
        AND t1.order_date = t2.order_date
        AND t1.category < t2.category
)
SELECT 
    category_a,
    category_b,
    COUNT(*) as co_occurrence_count,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT CONCAT(customer_id, order_date)) FROM transactions) as support_percent
FROM category_pairs
GROUP BY category_a, category_b
HAVING COUNT(*) >= 100
ORDER BY co_occurrence_count DESC;

-- Customer Churn Analysis
CREATE OR REPLACE VIEW customer_churn_analysis AS
SELECT 
    c.customer_id,
    c.customer_segment,
    c.total_orders,
    c.total_spent,
    DATEDIFF(CURDATE(), c.last_order_date) as days_since_last_order,
    CASE 
        WHEN DATEDIFF(CURDATE(), c.last_order_date) > 365 THEN 'High Risk'
        WHEN DATEDIFF(CURDATE(), c.last_order_date) > 180 THEN 'Medium Risk'
        WHEN DATEDIFF(CURDATE(), c.last_order_date) > 90 THEN 'Low Risk'
        ELSE 'Active'
    END as churn_risk,
    c.avg_order_value,
    c.customer_city,
    c.is_prime_member
FROM customers c
WHERE c.total_orders > 0
ORDER BY days_since_last_order DESC;

-- Seasonal Analysis
CREATE OR REPLACE VIEW seasonal_analysis AS
SELECT 
    td.season,
    t.order_month,
    td.month_name,
    COUNT(*) as total_orders,
    SUM(t.final_amount_inr) as total_revenue,
    AVG(t.final_amount_inr) as avg_order_value,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    SUM(CASE WHEN t.is_festival_sale = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100 as festival_sales_percent
FROM transactions t
JOIN time_dimension td ON DATE(t.order_date) = td.date_key
GROUP BY td.season, t.order_month, td.month_name
ORDER BY t.order_month;

-- =====================================================
-- DATA QUALITY AND MONITORING QUERIES
-- =====================================================

-- Daily Data Quality Metrics
CREATE OR REPLACE VIEW daily_data_quality AS
SELECT 
    DATE(t.order_date) as report_date,
    COUNT(*) as total_transactions,
    COUNT(CASE WHEN t.final_amount_inr IS NULL THEN 1 END) as null_amounts,
    COUNT(CASE WHEN t.customer_id IS NULL THEN 1 END) as null_customers,
    COUNT(CASE WHEN t.product_id IS NULL THEN 1 END) as null_products,
    AVG(t.final_amount_inr) as avg_order_value,
    MIN(t.final_amount_inr) as min_order_value,
    MAX(t.final_amount_inr) as max_order_value,
    COUNT(DISTINCT t.customer_id) as unique_customers,
    COUNT(DISTINCT t.product_id) as unique_products
FROM transactions t
WHERE t.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY DATE(t.order_date)
ORDER BY report_date DESC;
