# Cache-First Lead Generation Proxy

FastAPI application that caches and merges lead data from a provider waterfall.

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

### Provider chain

Searches are served by all configured providers and merged in order:
**Bytemine → Crustdata → GetLeads → Treg → ColdIQ → Wiza**. A provider only joins the chain if its key is
configured (for example `TREG_TOKEN`, `BYTEMINE_API_KEY`, or `WIZA_API_KEY`), so
adding a key is all it takes to put one in front. `SEARCH_PROVIDER` pins which one leads;
the others still follow it as fallbacks.

The order follows capability. Bytemine and Crustdata both return masked or
flagged contacts from search and charge on reveal, matching how the app bills.
Wiza's list workflow returns already-enriched contacts and spends an email
credit per search, so it sits last.

### Treg setup and customer billing

Treg is used only for lead generation through its routed
`treg.people.search` and `treg.people.enrich` capabilities. Set an org-scoped
`TREG_TOKEN`, `TREG_ORG_ID` (`4258` for the `bdotindustries` organization), and
optionally `SEARCH_PROVIDER=treg`. The token is
backend-only and must never be sent to a browser or model.

Every request that can reach Treg must carry a stable, non-email customer ID:

```http
X-Customer-ID: cust_8123
X-Workspace-ID: ws_9
Idempotency-Key: one-key-per-logical-request
```

The service writes those IDs to `X-Treg-Meta` itself, stores the returned
`X-Treg-Call-Id` and `X-Treg-Cost-Micro` in `treg_usage`, and refuses an
untagged Treg call before money can be spent. Cached responses incur no new
Treg usage.

For invoice generation, configure `BILLING_ADMIN_KEY` and call:

```http
GET /billing/treg/customers/cust_8123/usage?days=30
X-Billing-Admin-Key: <internal-admin-key>
```

That endpoint reads Treg's authoritative `usage/by-tag` ledger and verifies
that attributed plus unattributed spend equals the ledger total. Any nonzero
`unattributed_micro` is returned as a warning and should block invoice close.
Do not calculate invoices from the local audit table.

Customer spend controls are managed through:

```http
PUT /billing/treg/customers/cust_8123/budget
X-Billing-Admin-Key: <internal-admin-key>
Content-Type: application/json

{"daily_cap_micro": 5000000}
```

Use `{"status":"blocked"}` to stop Treg spend for a customer. Treg caps are
advisory under concurrent requests; the prepaid organization balance is the
hard limit.

A search moves to the next provider when the current one:

- **cannot express a filter** — Bytemine's contact search has no country or
  free-text keyword field, so those queries go to Crustdata rather than having
  the filter dropped;
- **fails** — including `402` when a provider's own credit balance is empty;
- **returns nothing** — the next provider may hold data this one lacks.

A `400` or `422` is about the query itself, so it surfaces immediately instead
of being retried against every provider. Every response reports which provider
served it and which were passed over:

```json
{
  "provider": "crustdata",
  "provider_attempts": [{ "provider": "bytemine", "outcome": "unsupported_filter", "detail": "keywords" }]
}
```

`GET /health` reports the live chain and flags a provider that was asked for but
has no key configured.

Crustdata searches are cursor-paginated and deduplicated. Keep one stable
`campaign_id` for a prospecting run, then pass each response's `next_cursor`
into the next request. The backend records every returned
`crustdata_person_id`, excludes profiles already seen by that campaign, and uses
a stable person-ID sort so pages do not drift.

```json
{
  "job_title": "Founder",
  "seniority": "owner",
  "location": "US",
  "industry": "computer software",
  "keywords": "cold email and outbound automation",
  "company_size": "1-10",
  "campaign_id": "august-founder-outreach",
  "limit": 10,
  "cursor": null
}
```

Response pagination fields:

```json
{
  "count": 10,
  "total": 1240,
  "next_cursor": "opaque-provider-cursor",
  "campaign_id": "august-founder-outreach"
}
```

- Do not progressively delete filters after a provider error. Unsupported
  filters now return `422` instead of being silently ignored.
- Title-only Crustdata searches return `422` by default. Set
  `allow_broad_search: true` only when the user deliberately requested a broad
  title search.
- `keywords` uses Crustdata hybrid semantic search and structured filters remain
  hard constraints; keywords are no longer treated as a second job title.
- Set `refresh: true` to bypass a cached page. Cache rows expire after
  `SEARCH_CACHE_TTL_SECONDS` (default `3600`; set `0` to disable caching).
- Pass `exclude_profiles` with profile URLs for one-off exclusions when a
  campaign ledger is not appropriate.
- Reset a campaign ledger with `DELETE /campaigns/{campaign_id}/seen`.

The frontend should stop on any non-2xx response and display the error. It must
not retry by stripping `location`, `industry`, `company_size`, `seniority`, or
semantic keywords.

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
