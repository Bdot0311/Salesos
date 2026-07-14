# Cache-First Lead Generation Proxy

FastAPI application that caches Wiza API results to reduce costs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Ensure PostgreSQL is running and create a database:
```sql
CREATE DATABASE leads_db;
```

4. Run the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /search
Search for leads (cache-first).

```json
{
  "job_title": "Software Engineer",
  "location": "San Francisco",
  "industry": "Technology",
  "company": "Google",
  "company_size": "10001+",
  "seniority": "senior",
  "limit": 10
}
```

Response includes `source: "cache"` or `source: "api"` to indicate data origin.

Runs Wiza's async 3-step list workflow (create → poll → enriched contacts) and
spends email/enrichment credits. Accepts a natural-language `query` (auto-parsed
into filters) as an alternative to structured fields.

### POST /prospects/preview
Fast, **synchronous** contact preview via Wiza `/prospects/search`. Returns the
total number of matching prospects plus up to 30 preview profiles — instantly and
**without spending email/enrichment credits**. Same request body as `/search`.
Use it to size an audience before committing to a full `/search`.

```json
{ "job_title": "VP of Sales", "industry": "SaaS", "company_size": "51-200", "limit": 20 }
```

Response: `{ "total": 412, "count": 20, "leads": [...] }`

### POST /company/search
Search for **companies** (not just people). Runs a prospect search with the
firmographic/company filters and rolls the results up into unique companies,
each with a few sample contacts. Cache-first. Same body as `/search`.

By default each unique company is **auto-enriched** with full firmographics
(industry, size, revenue, funding, socials) via Wiza company enrichment — **2
credits per company**, capped at `enrich_limit` (default 25, top matches first).
Disable with `?enrich=false` for a 0-credit name/domain roll-up.

```json
{ "industry": "Financial Services", "company_size": "201-500", "location": "New York" }
```

Query params: `?enrich=true|false` (default true), `?enrich_limit=25` (max 30).

Response: `{ "count": 12, "companies": [ { "company_name": "...", "company_domain": "...", "company_size": "201-500", "revenue_range": "...", "funding": "...", "matched_contacts": 4, "enriched": true, "sample_contacts": [...] } ] }`

### POST /company/enrich
**Domain search / company lookup.** Pass a `company_domain` (or name / LinkedIn)
and get back firmographics: industry, size, revenue, funding, location, socials.
Cache-first; each live lookup costs **2 Wiza API credits**.

```json
{ "company_domain": "stripe.com" }
```

Accepts any one of: `company_domain`, `company_name`, `company_linkedin_id`,
`company_linkedin_slug`.

### POST /enrich
Enrich a single person via Wiza Individual Reveal (linkedin_url, email, or
full_name + company/domain).

### POST /parse-icp
Parse a plain-English ICP description into structured Wiza search filters.

### GET /health
Health check.

### GET /cache/stats
View cache statistics.

### DELETE /cache
Clear all cached data.
