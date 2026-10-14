# Olist schema (DuckDB)

## Business rules
- Revenue = `order_items.price + order_items.freight_value`, delivered orders only (`orders.order_status = 'delivered'`).
- Count customers with `customers.customer_unique_id`, not `customer_id` (customer_id is per order).
- Trend analysis: exclude 2016 and Sept-Oct 2018 (incomplete months); use 2017-01 to 2018-08.
- Category names are Portuguese; join `category_translation` for English.
- Late = `orders.order_delivered_customer_date > orders.order_estimated_delivery_date` (timestamp comparison, not calendar date).

## Joins
- orders.customer_id -> customers.customer_id
- order_items.order_id / payments.order_id / reviews.order_id -> orders.order_id
- order_items.product_id -> products.product_id
- order_items.seller_id -> sellers.seller_id
- products.product_category_name -> category_translation.product_category_name

## Tables

### category_translation (71 rows)
| column | type |
|---|---|
| product_category_name | VARCHAR |
| product_category_name_english | VARCHAR |

### customers (99,441 rows)
| column | type |
|---|---|
| customer_id | VARCHAR |
| customer_unique_id | VARCHAR |
| customer_zip_code_prefix | VARCHAR |
| customer_city | VARCHAR |
| customer_state | VARCHAR |

### geolocation (1,000,163 rows)
| column | type |
|---|---|
| geolocation_zip_code_prefix | VARCHAR |
| geolocation_lat | DOUBLE |
| geolocation_lng | DOUBLE |
| geolocation_city | VARCHAR |
| geolocation_state | VARCHAR |

### order_items (112,650 rows)
| column | type |
|---|---|
| order_id | VARCHAR |
| order_item_id | BIGINT |
| product_id | VARCHAR |
| seller_id | VARCHAR |
| shipping_limit_date | TIMESTAMP |
| price | DOUBLE |
| freight_value | DOUBLE |

### orders (99,441 rows)
| column | type |
|---|---|
| order_id | VARCHAR |
| customer_id | VARCHAR |
| order_status | VARCHAR |
| order_purchase_timestamp | TIMESTAMP |
| order_approved_at | TIMESTAMP |
| order_delivered_carrier_date | TIMESTAMP |
| order_delivered_customer_date | TIMESTAMP |
| order_estimated_delivery_date | TIMESTAMP |

### payments (103,886 rows)
| column | type |
|---|---|
| order_id | VARCHAR |
| payment_sequential | BIGINT |
| payment_type | VARCHAR |
| payment_installments | BIGINT |
| payment_value | DOUBLE |

### products (32,951 rows)
| column | type |
|---|---|
| product_id | VARCHAR |
| product_category_name | VARCHAR |
| product_name_lenght | BIGINT |
| product_description_lenght | BIGINT |
| product_photos_qty | BIGINT |
| product_weight_g | BIGINT |
| product_length_cm | BIGINT |
| product_height_cm | BIGINT |
| product_width_cm | BIGINT |

### reviews (99,224 rows)
| column | type |
|---|---|
| review_id | VARCHAR |
| order_id | VARCHAR |
| review_score | BIGINT |
| review_comment_title | VARCHAR |
| review_comment_message | VARCHAR |
| review_creation_date | TIMESTAMP |
| review_answer_timestamp | TIMESTAMP |

### sellers (3,095 rows)
| column | type |
|---|---|
| seller_id | VARCHAR |
| seller_zip_code_prefix | VARCHAR |
| seller_city | VARCHAR |
| seller_state | VARCHAR |
