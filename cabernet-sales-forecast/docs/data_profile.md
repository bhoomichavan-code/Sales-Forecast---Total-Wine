# Data Profile -- Cabernet Sauvignon Internal Sales

- Rows: **66,974**  |  Columns (raw): 25
- Distinct stores: **269**  |  Distinct items: **1,451**  |  States: **28**

## Missing values
- `Retail`: 3 (0.00%)
- `Price_Band`: 3 (0.00%)
- `log_Retail`: 3 (0.00%)
- `Store_Tier_Matched`: 3 (0.00%)
- `Actual_Sales_$_L52W`: 2 (0.00%)
- `Normalized_Sales_$_L52W`: 2 (0.00%)

## Target: Normalized_Sales_$_L52W
- min $0 | p10 $198 | median $1,203 | mean $2,963 | p90 $6,249 | max $454,328
- Skew: **15.5** (extreme right skew -> model log scale)
- Exact-zero sales rows: 566 (0.8%)

## Key categoricals
- `Store_Size` (4 levels): Large=34,164, Medium=25,559, Small=6,826, Extra Large=425
- `Package_Type` (12 levels): 750ml=63,944, 1.5L=2,031, 375ml=597, 1L=225, 3L=112, 187ml-4p=22
- `Price_Band` (3 levels): Over_$50=29,286, Under_$20=21,945, $20-50=15,740, nan=3
- `Store_Tier` (5 levels): 3=17,460, 4=16,925, 2=16,320, 1=9,585, 0=6,684
