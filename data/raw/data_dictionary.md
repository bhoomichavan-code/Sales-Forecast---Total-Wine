# Data Dictionary — Internal Sales dataset

One row per **store × item** (Cabernet Sauvignon). Key = `PK`
(`Store_Number` + `_` + `Item_Code`).

| Column | Type | Definition |
|--------|------|-----------|
| PK | id | Store_Number + "_" + Item_Code |
| Store_Number | id | Store identifier |
| Store_State | cat | US state of the store |
| Item_Code | id | Internal item identifier |
| Item_Name | text | Product name |
| Package_Type | cat | Bottle size (e.g. 750ml, 1.5L) |
| Retail | $ | Shelf price |
| Actual_Sales_$_L52W | $ | Last-52-week sales dollars (**leakage** — not a predictor) |
| L52W_in_Stock | int | # weeks in the last year the item had inventory |
| Normalized_Sales_$_L52W | $ | **TARGET** = (L52W sales / weeks in stock) × 52 |
| Age_of_the_Store_(years) | float | Years since store opened |
| Store_Size | ordinal | Small / Medium / Large / Extra Large (selling space) |
| Households_(HH) | int | Households within a 5-mile radius |
| %_HH_Income_>_$100K | % | Share of households earning over $100K |
| Median_HH_Income | $ | Median household income (5-mi radius) |
| Average_Net_Worth | $ | Average household net worth (5-mi radius) |
| %_Population_w/_Bachelor's_Degree_+ | % | Share with a bachelor's degree or higher |
| %_Hispanic / %_Asian / %_African_American | % | Ethnicity shares (5-mi radius) |
| %_Population_Age_50-70 | % | Share of population aged 50–70 |
| Store_Tier_(Under_$20) / ($20-50) / (Over_$50) | ordinal 0–4 | Store sales-volume tier **within** that price band |
| Store_Tier | ordinal 0–4 | Overall store sales-volume tier (0 = lowest, 4 = highest) |

**Engineered in code** (`src/data_prep.py`): `Price_Band`, `Store_Tier_Matched`
(tier for the item's own price band), `Income_Age_Interaction`, `log_Retail`,
and out-of-fold target encodings `Item_Name_te`, `Store_Number_te`,
`Store_State_te`.
