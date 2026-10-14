Olist DuckDB tables (columns are VARCHAR unless typed):
category_translation(product_category_name, product_category_name_english)
customers(customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
geolocation(geolocation_zip_code_prefix, geolocation_lat DOUBLE, geolocation_lng DOUBLE, geolocation_city, geolocation_state)
order_items(order_id, order_item_id BIGINT, product_id, seller_id, shipping_limit_date TIMESTAMP, price DOUBLE, freight_value DOUBLE)
orders(order_id, customer_id, order_status, order_purchase_timestamp TIMESTAMP, order_approved_at TIMESTAMP, order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP, order_estimated_delivery_date TIMESTAMP)
payments(order_id, payment_sequential BIGINT, payment_type, payment_installments BIGINT, payment_value DOUBLE)
products(product_id, product_category_name, product_name_lenght BIGINT, product_description_lenght BIGINT, product_photos_qty BIGINT, product_weight_g BIGINT, product_length_cm BIGINT, product_height_cm BIGINT, product_width_cm BIGINT)
reviews(review_id, order_id, review_score BIGINT, review_comment_title, review_comment_message, review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP)
sellers(seller_id, seller_zip_code_prefix, seller_city, seller_state)
Rules:
- Revenue = order_items.price + freight_value; delivered orders only (order_status='delivered').
- Count customers by customer_unique_id (customer_id is per order).
- Trends: use 2017-01..2018-08 (2016, 2018-09/10 incomplete).
- Category names are Portuguese; English via category_translation.
- Late = order_delivered_customer_date > order_estimated_delivery_date (timestamps).
Joins: orders.customer_id=customers.customer_id; order_items/payments/reviews.order_id=orders.order_id;
order_items.product_id=products.product_id; order_items.seller_id=sellers.seller_id;
products.product_category_name=category_translation.product_category_name
