# Data Quality Rules

Rules applied to procurement data before it enters the forecasting pipeline.

---

## 1. Schema Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| SCH-001 | All | All required columns must be present: `product_id`, `product_name`, `category`, `quantity`, `unit_price`, `supplier`, `order_date`, `delivery_date`, `status` | Reject entire file |
| SCH-002 | All | No additional unexpected columns that could shadow required ones | Log warning |

---

## 2. Completeness Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| CMP-001 | `product_id` | Must not be null | Drop row, log error |
| CMP-002 | `supplier` | Must not be null or empty string | Drop row, log error |
| CMP-003 | `order_date` | Must not be null | Drop row, log error |
| CMP-004 | `quantity` | Must not be null | Drop row, log error |
| CMP-005 | `unit_price` | Must not be null | Drop row, log error |
| CMP-006 | `delivery_date` | Null allowed (order not yet delivered); flag as `pending_delivery` | Flag row |
| CMP-007 | `status` | Must not be null | Fill with `"Unknown"`, log warning |

---

## 3. Format / Type Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| FMT-001 | `product_id` | Must match pattern `P\d{3,}` (e.g. `P001`) | Drop row, log error |
| FMT-002 | `order_date` | Must be parseable as ISO-8601 date (`YYYY-MM-DD`) | Drop row, log error |
| FMT-003 | `delivery_date` | Must be parseable as ISO-8601 date (`YYYY-MM-DD`) when not null | Drop row, log error |
| FMT-004 | `quantity` | Must be a non-negative integer | Drop row, log error |
| FMT-005 | `unit_price` | Must be a non-negative float | Drop row, log error |
| FMT-006 | `status` | Must be one of `Delivered`, `In Transit`, `Pending`, `Unknown` | Coerce to `"Unknown"`, log warning |

---

## 4. Range / Boundary Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| RNG-001 | `quantity` | Must be > 0 | Drop row, log error |
| RNG-002 | `unit_price` | Must be > 0.00 | Drop row, log error |
| RNG-003 | `order_date` | Must not be in the future (> today) | Drop row, log error |
| RNG-004 | `delivery_date` | Must be ≥ `order_date` when not null | Drop row, log error |
| RNG-005 | `unit_price` | Must be ≤ 100,000 (sanity cap) | Flag for manual review |

---

## 5. Uniqueness Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| UNQ-001 | `product_id` + `order_date` | Combination must be unique per record (no duplicate orders) | Keep first occurrence, drop subsequent, log warning |

---

## 6. Supplier Normalization Rules

| Rule ID | Column | Rule | Action on Failure |
|---------|--------|------|-------------------|
| SUP-001 | `supplier` | Name must appear in the canonical supplier list after normalization | Flag for review; do not drop |
| SUP-002 | `supplier` | Extra whitespace must be stripped | Auto-correct |
| SUP-003 | `supplier` | Legal suffixes must use canonical short form (`Inc`, `Ltd`, `Corp`, `Co`, `LLC`) | Auto-correct via `supplier_normalization.py` |
| SUP-004 | `supplier` | Title-case must be applied (e.g. `acme corp` → `Acme Corp`) | Auto-correct |
| SUP-005 | `supplier` | Known aliases must be mapped to their canonical name (see `SUPPLIER_ALIAS_MAP` in `app/cleaning/supplier_normalization.py`) | Auto-correct |

### Canonical Supplier List

| Canonical Name | Common Aliases |
|----------------|----------------|
| Acme Corp | acme, acme corporation, acme corp. |
| TechSupply Inc | techsupply, tech supply inc, techsupply incorporated |
| Global Parts Co | global parts, global parts company, globalparts co |
| ToolMaster Ltd | toolmaster, tool master ltd, tool master limited |
| RawSource Inc | rawsource, raw source inc, raw source |

---

## 7. Cross-Column Consistency Rules

| Rule ID | Columns | Rule | Action on Failure |
|---------|---------|------|-------------------|
| CRS-001 | `status`, `delivery_date` | If `status == "Delivered"` then `delivery_date` must not be null | Log warning, flag row |
| CRS-002 | `status`, `delivery_date` | If `status == "Pending"` then `delivery_date` may be null | No action |
| CRS-003 | `quantity`, `unit_price` | Derived `total_value = quantity × unit_price` must be > 0 | Drop row, log error |

---

## 8. Enforcement Pipeline Order

Rules are enforced in the following order during ingestion/cleaning:

1. **SCH** — schema check (fail-fast; reject file if schema is wrong)
2. **CMP** — completeness checks
3. **FMT** — format / type coercions
4. **RNG** — range / boundary validation
5. **UNQ** — deduplication
6. **SUP** — supplier normalization (`app/cleaning/supplier_normalization.py`)
7. **CRS** — cross-column consistency

---

## 9. Reporting

After each pipeline run, a quality report must be emitted containing:

- Total records loaded
- Records dropped (with rule ID and count per rule)
- Records auto-corrected (with rule ID and count per rule)
- Records flagged for manual review
- Supplier normalization change log (original → canonical)
- Final clean record count

Reports are written to `data/quality_reports/` as JSON files with a UTC timestamp in the filename (e.g. `quality_report_20260507T120000Z.json`).
