-- Data Loading and Validation Procedures
-- Amazon India Analytics Platform

-- =====================================================
-- DATA LOADING PROCEDURES
-- =====================================================

-- Load Transactions Data with Validation
DELIMITER //
CREATE PROCEDURE LoadTransactionData()
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    -- Validate data before loading
    INSERT INTO transactions (
        transaction_id, customer_id, product_id, order_date,
        order_year, order_month, order_quarter,
        category, brand, product_name,
        original_price_inr, discount_percent, final_amount_inr,
        customer_city, customer_state, age_group, is_prime_member,
        payment_method, delivery_days, customer_rating, product_rating,
        is_festival_sale, festival_name
    )
    SELECT 
        transaction_id,
        customer_id,
        product_id,
        STR_TO_DATE(order_date, '%Y-%m-%d') as order_date,
        YEAR(STR_TO_DATE(order_date, '%Y-%m-%d')) as order_year,
        MONTH(STR_TO_DATE(order_date, '%Y-%m-%d')) as order_month,
        QUARTER(STR_TO_DATE(order_date, '%Y-%m-%d')) as order_quarter,
        category,
        brand,
        product_name,
        CAST(original_price_inr as DECIMAL(10,2)),
        CAST(discount_percent as DECIMAL(5,2)),
        CAST(final_amount_inr as DECIMAL(10,2)),
        customer_city,
        customer_state,
        age_group,
        CASE WHEN is_prime_member = 'True' THEN 1 ELSE 0 END,
        payment_method,
        CAST(delivery_days as SIGNED),
        CAST(customer_rating as DECIMAL(2,1)),
        CAST(product_rating as DECIMAL(2,1)),
        CASE WHEN is_festival_sale = 'True' THEN 1 ELSE 0 END,
        festival_name
    FROM temp_transactions_import
    WHERE transaction_id IS NOT NULL
      AND customer_id IS NOT NULL
      AND product_id IS NOT NULL
      AND order_date IS NOT NULL
      AND final_amount_inr > 0;
    
    COMMIT;
    
    SELECT 
        COUNT(*) as records_loaded,
        MIN(order_date) as earliest_date,
        MAX(order_date) as latest_date,
        SUM(final_amount_inr) as total_revenue
    FROM transactions;
    
END //
DELIMITER ;

-- Load Products Data
DELIMITER //
CREATE PROCEDURE LoadProductsData()
BEGIN
    INSERT IGNORE INTO products (
        product_id, product_name, category, subcategory, brand,
        base_price_2015, avg_rating, is_prime_eligible, launch_year
    )
    SELECT 
        product_id,
        product_name,
        category,
        subcategory,
        brand,
        CAST(base_price_2015 as DECIMAL(10,2)),
        CAST(rating as DECIMAL(2,1)),
        CASE WHEN is_prime_eligible = 'True' THEN 1 ELSE 0 END,
        CAST(launch_year as SIGNED)
    FROM temp_products_import
    WHERE product_id IS NOT NULL;
    
    SELECT COUNT(*) as products_loaded FROM products;
END //
DELIMITER ;

-- Generate Time Dimension Data
DELIMITER //
CREATE PROCEDURE GenerateTimeDimension(IN start_date DATE, IN end_date DATE)
BEGIN
    DECLARE current_date DATE DEFAULT start_date;
    
    WHILE current_date <= end_date DO
        INSERT IGNORE INTO time_dimension (
            date_key, year, quarter, month, month_name, day, day_name,
            day_of_year, week_of_year, is_weekend, season,
            fiscal_year, fiscal_quarter
        ) VALUES (
            current_date,
            YEAR(current_date),
            QUARTER(current_date),
            MONTH(current_date),
            MONTHNAME(current_date),
            DAY(current_date),
            DAYNAME(current_date),
            DAYOFYEAR(current_date),
            WEEK(current_date),
            CASE WHEN DAYOFWEEK(current_date) IN (1,7) THEN 1 ELSE 0 END,
            CASE 
                WHEN MONTH(current_date) IN (12,1,2) THEN 'Winter'
                WHEN MONTH(current_date) IN (3,4,5) THEN 'Spring'
                WHEN MONTH(current_date) IN (6,7,8) THEN 'Summer'
                ELSE 'Autumn'
            END,
            CASE 
                WHEN MONTH(current_date) >= 4 THEN YEAR(current_date)
                ELSE YEAR(current_date) - 1
            END,
            CASE 
                WHEN MONTH(current_date) IN (4,5,6) THEN 1
                WHEN MONTH(current_date) IN (7,8,9) THEN 2
                WHEN MONTH(current_date) IN (10,11,12) THEN 3
                ELSE 4
            END
        );
        
        SET current_date = DATE_ADD(current_date, INTERVAL 1 DAY);
    END WHILE;
    
    SELECT COUNT(*) as dates_generated FROM time_dimension;
END //
DELIMITER ;

-- Update Customer Master Data
DELIMITER //
CREATE PROCEDURE UpdateCustomerMaster()
BEGIN
    INSERT INTO customers (
        customer_id, customer_city, customer_state, age_group,
        is_prime_member, first_order_date, last_order_date,
        total_orders, total_spent, avg_order_value
    )
    SELECT 
        customer_id,
        MAX(customer_city) as customer_city,
        MAX(customer_state) as customer_state,
        MAX(age_group) as age_group,
        MAX(is_prime_member) as is_prime_member,
        MIN(order_date) as first_order_date,
        MAX(order_date) as last_order_date,
        COUNT(*) as total_orders,
        SUM(final_amount_inr) as total_spent,
        AVG(final_amount_inr) as avg_order_value
    FROM transactions
    GROUP BY customer_id
    ON DUPLICATE KEY UPDATE
        last_order_date = VALUES(last_order_date),
        total_orders = VALUES(total_orders),
        total_spent = VALUES(total_spent),
        avg_order_value = VALUES(avg_order_value),
        updated_at = CURRENT_TIMESTAMP;
    
    -- Update customer segments
    UPDATE customers SET customer_segment = 
        CASE 
            WHEN total_spent > 100000 AND total_orders > 20 THEN 'VIP'
            WHEN total_spent > 50000 AND total_orders > 10 THEN 'Premium'
            WHEN total_spent > 10000 OR total_orders > 5 THEN 'Regular'
            ELSE 'New'
        END;
        
    SELECT COUNT(*) as customers_updated FROM customers;
END //
DELIMITER ;

-- Data Quality Check
DELIMITER //
CREATE PROCEDURE DataQualityCheck()
BEGIN
    SELECT 'Data Quality Report' as report_type;
    
    -- Transaction data quality
    SELECT 
        'Transactions' as table_name,
        COUNT(*) as total_records,
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(DISTINCT product_id) as unique_products,
        SUM(CASE WHEN final_amount_inr IS NULL THEN 1 ELSE 0 END) as null_amounts,
        SUM(CASE WHEN order_date IS NULL THEN 1 ELSE 0 END) as null_dates,
        MIN(order_date) as earliest_date,
        MAX(order_date) as latest_date,
        SUM(final_amount_inr) as total_revenue
    FROM transactions
    
    UNION ALL
    
    -- Product data quality
    SELECT 
        'Products' as table_name,
        COUNT(*) as total_records,
        COUNT(DISTINCT category) as unique_categories,
        COUNT(DISTINCT brand) as unique_brands,
        SUM(CASE WHEN product_name IS NULL THEN 1 ELSE 0 END) as null_names,
        SUM(CASE WHEN category IS NULL THEN 1 ELSE 0 END) as null_categories,
        0 as earliest_date,
        0 as latest_date,
        0 as total_revenue
    FROM products
    
    UNION ALL
    
    -- Customer data quality
    SELECT 
        'Customers' as table_name,
        COUNT(*) as total_records,
        COUNT(DISTINCT customer_city) as unique_cities,
        COUNT(DISTINCT customer_state) as unique_states,
        SUM(CASE WHEN customer_city IS NULL THEN 1 ELSE 0 END) as null_cities,
        SUM(CASE WHEN age_group IS NULL THEN 1 ELSE 0 END) as null_age_groups,
        0 as earliest_date,
        0 as latest_date,
        SUM(total_spent) as total_revenue
    FROM customers;
    
END //
DELIMITER ;
