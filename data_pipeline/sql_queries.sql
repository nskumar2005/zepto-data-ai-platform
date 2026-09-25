-- 1. SELECT / WHERE
SELECT title, price_gbp, rating FROM books WHERE price_gbp > 30;

-- 2. ORDER BY
SELECT title, price_gbp FROM books ORDER BY price_gbp DESC;

-- 3. LIMIT
SELECT title, rating FROM books ORDER BY rating DESC, title LIMIT 10;

-- 4. DISTINCT
SELECT DISTINCT rating FROM books ORDER BY rating;

-- 5. BETWEEN
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 ORDER BY price_gbp;

-- 6. JOIN
SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, c.category_name, b.title
LIMIT 10;
