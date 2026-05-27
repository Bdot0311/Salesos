"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Prospeo API results to reduce costs.
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from sqlalchemy import Column, String, Text, DateTime, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    database_url: str
    prospeo_api_key: str
    anthropic_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

    @property
    def async_database_url(self) -> str:
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()

PROSPEO_BASE = "https://api.prospeo.io"


# =============================================================================
# Prospeo Filter Mappings
# =============================================================================

# Our seniority keys -> Prospeo seniority enum values
SENIORITY_MAP = {
    "entry":    "Entry",
    "training": "Entry",
    "intern":   "Intern",
    "junior":   "Entry",
    "senior":   "Senior",
    "manager":  "Manager",
    "head":     "Head",
    "director": "Director",
    "partner":  "Partner",
    "vp":       "Vice President",
    "c_suite":  "C-Suite",
    "cxo":      "C-Suite",
    "owner":    "Founder/Owner",
    "founder":  "Founder/Owner",
}

# Our size keys -> Prospeo headcount_range values (one-to-many for broad bands)
COMPANY_SIZE_MAP: dict[str, list[str]] = {
    "1-10":       ["1-10"],
    "11-50":      ["11-20", "21-50"],
    "51-200":     ["51-100", "101-200"],
    "201-500":    ["201-500"],
    "501-1000":   ["501-1000"],
    "1001-5000":  ["1001-2000", "2001-5000"],
    "5001-10000": ["5001-10000"],
    "10001+":     ["10000+"],
}

# Buyer intent news categories supported by Prospeo company_news filter
VALID_NEWS_CATEGORIES = {
    "Mergers", "Funding", "Product Launch", "Hiring", "Partnership",
    "Expansion", "Award", "Leadership Change", "IPO",
}

# Prospeo department enum values
VALID_DEPARTMENTS = {
    "C-Suite", "Consulting", "Design", "Education & Coaching",
    "Engineering & Technical", "Finance", "Human Resources",
    "Information Technology", "Legal", "Marketing", "Medical & Health",
    "Operations", "Product", "Sales",
}

# Location type detection
_COUNTRIES = {"united states", "us", "usa", "united kingdom", "uk", "canada",
              "australia", "germany", "france", "india", "china", "japan",
              "brazil", "israel", "singapore", "netherlands", "spain", "italy",
              "sweden", "norway", "denmark", "finland", "mexico", "south korea",
              "new zealand", "ireland", "switzerland", "austria", "belgium",
              "portugal", "poland", "czech republic", "ukraine", "russia"}
_US_STATES = {"alabama", "alaska", "arizona", "arkansas", "california", "colorado",
              "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
              "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
              "maine", "maryland", "massachusetts", "michigan", "minnesota",
              "mississippi", "missouri", "montana", "nebraska", "nevada",
              "new hampshire", "new jersey", "new mexico", "new york",
              "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
              "pennsylvania", "rhode island", "south carolina", "south dakota",
              "tennessee", "texas", "utah", "vermont", "virginia", "washington",
              "west virginia", "wisconsin", "wyoming"}


def normalize_location(location: str) -> str:
    """Return canonical location string for Prospeo include/exclude arrays."""
    loc = location.strip()
    loc_lower = loc.lower()
    if loc_lower in ("us", "usa"):
        return "United States"
    if loc_lower == "uk":
        return "United Kingdom"
    return loc.title() if loc_lower in _US_STATES else loc


# =============================================================================
# Database Models
# =============================================================================

class Base(DeclarativeBase):
    pass


class CachedSearch(Base):
    """Stores search parameters and their hash for quick lookup."""
    __tablename__ = "cached_searches"

    search_hash = Column(String(64), primary_key=True)
    search_params = Column(Text, nullable=False)
    results = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =============================================================================
# Database Setup
# =============================================================================

engine = create_async_engine(settings.async_database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class SearchRequest(BaseModel):
    # Plain-text ICP query (auto-parsed if no structured filters provided)
    query: Optional[str] = None

    # Contact filters
    job_title: Optional[str] = None
    departments: Optional[list[str]] = None
    seniority: Optional[str] = None
    location: Optional[str] = None          # person location
    company_location: Optional[str] = None  # company HQ location

    # Company filters
    company: Optional[str] = None
    company_domain: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    technologies: Optional[list[str]] = None
    keywords: Optional[str] = None

    # Buyer intent — news/event categories that signal purchase readiness
    # Valid values: Mergers, Funding, Product Launch, Hiring, Partnership,
    #               Expansion, Award, Leadership Change, IPO
    intent_topics: Optional[list[str]] = None
    # How far back to look for intent events (60, 90, 180, or 365 days)
    intent_days: int = 90

    # Revenue range (USD)
    revenue_min: Optional[int] = None
    revenue_max: Optional[int] = None

    # Career change signal
    job_change_days: Optional[int] = None   # e.g. 90 — contacts who changed jobs in last N days

    # Legacy signals field (mapped to job_change_days if present)
    signals: Optional[list[str]] = None
    signals_since_days: int = 90

    limit: int = 10

    @field_validator("keywords", "seniority", "job_title", "location", "company_location",
                     "company", "company_domain", "industry", "company_size", "query", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return v

    @field_validator("technologies", "departments", "intent_topics", mode="before")
    @classmethod
    def coerce_list(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


class ICPParseRequest(BaseModel):
    text: str


class SearchResponse(BaseModel):
    success: bool = True
    source: str
    from_cache: bool = False
    count: int
    total: int
    leads: list
    data: list


# =============================================================================
# Transform
# =============================================================================

def transform_prospeo_contact(record: dict, search_params: dict = None) -> dict:
    """Transform a Prospeo enriched record into the internal lead format."""
    search_params = search_params or {}

    person = record.get("person", record)
    company = record.get("company", {})

    email_obj = person.get("email") or {}
    email = email_obj.get("email") if isinstance(email_obj, dict) else email_obj

    mobile_obj = person.get("mobile") or {}
    mobile = mobile_obj.get("mobile") if isinstance(mobile_obj, dict) else mobile_obj

    loc = person.get("location") or {}

    # Pull the current job title from job_history if top-level missing
    job_title = person.get("current_job_title")
    if not job_title:
        for job in (person.get("job_history") or []):
            if job.get("current"):
                job_title = job.get("title")
                break

    # Funding / buyer intent context
    funding = company.get("funding") or {}
    news = record.get("news") or []

    return {
        "contact_name": person.get("full_name"),
        "first_name": person.get("first_name"),
        "last_name": person.get("last_name"),
        "job_title": job_title,
        "linkedin_url": person.get("linkedin_url"),
        "business_email": email,
        "email_status": email_obj.get("status") if isinstance(email_obj, dict) else None,
        "phone": mobile,
        "location_city": loc.get("city"),
        "location_state": loc.get("state"),
        "location_country": loc.get("country"),
        "company_name": company.get("name"),
        "company_domain": company.get("domain") or company.get("website"),
        "company_linkedin_url": company.get("linkedin_url"),
        "industry": company.get("industry") or search_params.get("industry"),
        "company_size": company.get("employee_range") or search_params.get("company_size"),
        "company_revenue_range": company.get("revenue_range"),
        "technologies": search_params.get("technologies"),
        "funding_stage": funding.get("last_funding_type"),
        "funding_total": funding.get("total_funding_amount"),
        "intent_signals": news or None,
        "skills": person.get("skills"),
        "raw_data": record,
    }


def generate_search_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


# =============================================================================
# Prospeo Two-Step Workflow: Search → Enrich
# =============================================================================

async def fetch_from_prospeo(params: dict) -> list:
    """
    Two-step Prospeo workflow:
      1. POST /search-person  — find contacts with buyer intent + standard filters
      2. POST /enrich-person  — enrich each result by person_id (concurrent)
    """
    # Map legacy signals list to job_change_days
    if params.get("signals") and not params.get("job_change_days"):
        if any(s in ("job_change", "promotion", "companyChange") for s in params["signals"]):
            params["job_change_days"] = params.get("signals_since_days", 90)

    searchable_fields = (
        "job_title", "departments", "location", "company_location", "industry",
        "company", "company_domain", "company_size", "seniority", "technologies",
        "intent_topics", "keywords", "revenue_min", "revenue_max", "job_change_days",
    )
    if not any(params.get(k) for k in searchable_fields):
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    headers = {
        "X-KEY": settings.prospeo_api_key,
        "Content-Type": "application/json",
    }

    # ------------------------------------------------------------------
    # Step 1: Build search filters
    # ------------------------------------------------------------------
    filters: dict = {}

    # Job title free-text
    if params.get("job_title"):
        filters["person_job_title"] = {"include": [params["job_title"]]}

    # Department
    if params.get("departments"):
        valid_depts = [d for d in params["departments"] if d in VALID_DEPARTMENTS]
        if valid_depts:
            filters["person_department"] = {"include": valid_depts}

    # Seniority
    if params.get("seniority"):
        level = SENIORITY_MAP.get(params["seniority"].lower())
        if level:
            filters["person_seniority"] = {"include": [level]}

    # Person location
    if params.get("location"):
        filters["person_location_search"] = {
            "include": [normalize_location(params["location"])]
        }

    # Company HQ location
    if params.get("company_location"):
        filters["company_location_search"] = {
            "include": [normalize_location(params["company_location"])]
        }

    # Company name / domain
    if params.get("company") or params.get("company_domain"):
        company_filter: dict = {}
        if params.get("company"):
            company_filter["names"] = {"include": [params["company"]]}
        if params.get("company_domain"):
            company_filter["websites"] = {"include": [params["company_domain"]]}
        filters["company"] = company_filter

    # Industry
    if params.get("industry"):
        filters["company_industry"] = {"include": [params["industry"]]}

    # Company headcount
    if params.get("company_size"):
        ranges = COMPANY_SIZE_MAP.get(params["company_size"], [])
        if ranges:
            filters["company_headcount_range"] = {"include": ranges}

    # Technologies (buyer signal + stack filter)
    if params.get("technologies"):
        filters["company_technology"] = {"include": params["technologies"][:20]}

    # Free-text keyword search
    if params.get("keywords"):
        filters["person_name_or_job_title"] = params["keywords"]

    # Revenue range → custom headcount proxy (Prospeo doesn't have a direct
    # revenue filter; store it for enrichment-time filtering)
    # We pass it through but don't add it to Prospeo filters directly.

    # ---- Buyer Intent: company_news ----
    if params.get("intent_topics"):
        valid_cats = [t for t in params["intent_topics"] if t in VALID_NEWS_CATEGORIES]
        if valid_cats:
            days = params.get("intent_days", 90)
            if days not in (60, 90, 180, 365):
                days = 90
            filters["company_news"] = {
                "categories": {"include": valid_cats},
                "days": days,
            }

    # ---- Career signal: job change ----
    if params.get("job_change_days"):
        filters["person_job_change"] = {"days": params["job_change_days"]}

    # Require contacts to have email available
    filters["person_contact_details"] = {"email": True}

    search_body = {
        "filters": filters,
        "page": 1,
    }

    print(f"Prospeo search body: {json.dumps(search_body)}")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # ---- Step 1: Search ----
        search_resp = None
        for attempt in range(3):
            search_resp = await client.post(
                f"{PROSPEO_BASE}/search-person",
                headers=headers,
                json=search_body,
            )
            print(f"Prospeo search status: {search_resp.status_code}")
            if search_resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Rate limited — retrying in {wait}s (attempt {attempt + 1}/3)")
            await asyncio.sleep(wait)

        print(f"Prospeo search response: {search_resp.text[:400]}")

        if search_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Prospeo rate limit reached — please try again in a moment")
        if search_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=search_resp.status_code,
                detail=f"Prospeo search error: {search_resp.text}",
            )

        search_data = search_resp.json()
        if search_data.get("error"):
            return []

        results = search_data.get("results", [])
        print(f"Prospeo search returned {len(results)} results")

        if not results:
            return []

        # Trim to requested limit before enriching (costs credits per enrich)
        limit = min(params.get("limit", 10), len(results))
        results = results[:limit]

        # ---- Step 2: Enrich each person concurrently ----
        async def enrich_one(person_record: dict) -> dict | None:
            person = person_record.get("person", {})
            person_id = person.get("person_id")
            if not person_id:
                return person_record  # return search data as-is if no ID

            for attempt in range(3):
                r = await client.post(
                    f"{PROSPEO_BASE}/enrich-person",
                    headers=headers,
                    json={
                        "data": {"person_id": person_id},
                        "only_verified_email": True,
                    },
                )
                print(f"Prospeo enrich status ({person_id}): {r.status_code}")
                if r.status_code == 429:
                    wait = 2 ** attempt
                    print(f"Rate limited on enrich — retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if r.status_code in (200, 201):
                    enriched = r.json()
                    if not enriched.get("error"):
                        # Merge company data from search result into enriched record
                        if "company" not in enriched and "company" in person_record:
                            enriched["company"] = person_record["company"]
                        return enriched
                break
            # Enrichment failed — fall back to search-only record
            return person_record

        # Run all enrichments concurrently (Prospeo is synchronous, no polling needed)
        enriched_results = await asyncio.gather(*[enrich_one(r) for r in results])
        return [r for r in enriched_results if r is not None]


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches Prospeo API results to reduce costs",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try:
        await init_db()
    except Exception as e:
        print(f"WARNING: Database initialization failed: {e}")
        print("App will continue starting — DB will be retried on first request.")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# =============================================================================
# ICP Parser Endpoint
# =============================================================================

@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """
    Parse a plain-English ICP description into structured Prospeo search filters.

    Example input:
      "CTOs at fintech startups using Salesforce, 50-200 employees in NYC,
       companies that recently got funding or had a leadership change"

    Returns structured filters ready to pass directly into POST /search.
    """
    if not settings.anthropic_api_key:
        # No LLM available — return the raw text as a keyword fallback
        return {"success": True, "filters": {"keywords": request.text}}

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""You are an ICP (Ideal Customer Profile) parser for a B2B lead generation platform powered by Prospeo.

Parse the following description into structured search filters. Return ONLY valid JSON with these fields (omit fields not mentioned or not clearly implied):

{{
  "job_title": "string — e.g. CTO, VP of Sales",
  "departments": ["array — valid values: C-Suite, Consulting, Design, Education & Coaching, Engineering & Technical, Finance, Human Resources, Information Technology, Legal, Marketing, Medical & Health, Operations, Product, Sales"],
  "seniority": "one of: entry, intern, junior, senior, head, manager, director, partner, vp, c_suite, owner, founder",
  "location": "person's city, state, or country e.g. San Francisco",
  "company_location": "company HQ city, state, or country",
  "company": "specific company name if mentioned",
  "company_domain": "company website domain e.g. acme.com",
  "company_size": "one of: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+",
  "industry": "industry string e.g. Software Development, Financial Services, Healthcare",
  "technologies": ["tech stack strings e.g. Salesforce, HubSpot, AWS"],
  "keywords": "additional free-text search terms",
  "intent_topics": ["buyer intent categories — only from: Mergers, Funding, Product Launch, Hiring, Partnership, Expansion, Award, Leadership Change, IPO"],
  "intent_days": integer — one of 60, 90, 180, 365 (lookback window for intent events),
  "revenue_min": integer in USD,
  "revenue_max": integer in USD,
  "job_change_days": integer — e.g. 90, if recently changed jobs is mentioned
}}

ICP Description: {request.text}

Return only the JSON object, no explanation, no markdown fences."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"ICP parser returned invalid JSON: {raw}")

    return {"success": True, "filters": parsed}


# =============================================================================
# Search Endpoint
# =============================================================================

@app.post("/search", response_model=SearchResponse)
async def search_leads(request: SearchRequest):
    """
    Cache-first lead search with buyer intent.

    Flow:
    1. Check DB cache — return immediately if found
    2. Cache miss:
       a. POST /search-person (with company_news intent filter + standard filters)
       b. POST /enrich-person for each result concurrently (gets verified email + mobile)
    3. Cache raw enriched results
    4. Return transformed leads
    """
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != "" and v != []}

    print(f"=== SEARCH REQUEST ===")
    print(f"Filtered params: {params}")

    # Auto-parse plain-text query into structured filters
    structured_fields = {
        "job_title", "departments", "seniority", "location", "company_location",
        "company", "company_domain", "company_size", "industry", "technologies",
        "keywords", "intent_topics", "revenue_min", "revenue_max", "job_change_days",
    }
    raw_query = params.pop("query", None)
    if raw_query and not any(params.get(f) for f in structured_fields):
        parsed_filters: dict = {}

        if settings.anthropic_api_key:
            try:
                print(f"Auto-parsing query via ICP parser: {raw_query}")
                icp_result = await parse_icp(ICPParseRequest(text=raw_query))
                parsed_filters = icp_result.get("filters", {})
                print(f"ICP parser returned: {parsed_filters}")
            except Exception as e:
                print(f"WARNING: ICP parser failed ({e}) — falling back to keyword search")

        # Merge parsed filters into params
        for k, v in parsed_filters.items():
            if v is not None and v != "" and v != []:
                params[k] = v

        # If parsing produced nothing useful, use the raw query as a keyword search
        if not any(params.get(f) for f in structured_fields):
            print(f"ICP parse yielded no filters — using raw query as keyword search")
            params["keywords"] = raw_query

        print(f"Final params after query resolution: {params}")

    search_hash = generate_search_hash(params)
    print(f"Search hash: {search_hash}")

    async with async_session() as session:
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached:
            print(f"Cache HIT for hash: {search_hash}")
            data = json.loads(cached.results)
            leads = [transform_prospeo_contact(r, params) for r in data]
            return SearchResponse(
                success=True, source="cache", from_cache=True,
                count=len(leads), total=len(leads), leads=leads, data=data,
            )

        print(f"Cache MISS — calling Prospeo with params: {params}")
        raw_results = await fetch_from_prospeo(params)
        print(f"Prospeo returned {len(raw_results)} enriched leads")

        leads = [transform_prospeo_contact(r, params) for r in raw_results]

        new_cache = CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(params),
            results=json.dumps(raw_results),
        )
        session.add(new_cache)
        await session.commit()

        return SearchResponse(
            success=True, source="api", from_cache=False,
            count=len(leads), total=len(leads), leads=leads, data=raw_results,
        )


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@app.delete("/cache")
async def clear_cache():
    async with async_session() as session:
        await session.execute(CachedSearch.__table__.delete())
        await session.commit()
    return {"message": "Cache cleared"}


@app.delete("/cache/empty")
async def clear_empty_cache():
    async with async_session() as session:
        stmt = select(CachedSearch)
        result = await session.execute(stmt)
        all_cached = result.scalars().all()
        deleted = 0
        for cached in all_cached:
            if len(json.loads(cached.results)) == 0:
                await session.delete(cached)
                deleted += 1
        await session.commit()
    return {"message": f"Cleared {deleted} empty cached searches"}


@app.get("/cache/stats")
async def cache_stats():
    async with async_session() as session:
        from sqlalchemy import func, desc
        count = (await session.execute(select(func.count()).select_from(CachedSearch))).scalar()
        recent = (await session.execute(
            select(CachedSearch).order_by(desc(CachedSearch.created_at)).limit(10)
        )).scalars().all()

        return {
            "cached_searches": count,
            "recent_searches": [
                {
                    "search_hash": s.search_hash,
                    "params": json.loads(s.search_params),
                    "result_count": len(json.loads(s.results)),
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in recent
            ],
        }


@app.get("/cache/all")
async def get_all_cached_leads():
    """Retrieve all cached leads (fallback when Prospeo credits are exhausted)."""
    async with async_session() as session:
        stmt = select(CachedSearch).order_by(CachedSearch.created_at.desc())
        all_cached = (await session.execute(stmt)).scalars().all()

        all_leads = []
        seen = set()

        for cached in all_cached:
            data = json.loads(cached.results)
            search_params = json.loads(cached.search_params)
            for record in data:
                person = record.get("person", record)
                key = f"{person.get('full_name', '')}-{record.get('company', {}).get('name', '')}"
                if key not in seen:
                    seen.add(key)
                    all_leads.append(transform_prospeo_contact(record, search_params))

        return {
            "success": True, "from_cache": True,
            "leads": all_leads, "count": len(all_leads), "total": len(all_leads),
            "message": "All cached leads retrieved",
        }


@app.get("/cache/search/{search_hash}")
async def get_cached_search(search_hash: str):
    async with async_session() as session:
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        cached = (await session.execute(stmt)).scalar_one_or_none()
        if not cached:
            raise HTTPException(status_code=404, detail="Cached search not found")

        data = json.loads(cached.results)
        search_params = json.loads(cached.search_params)
        leads = [transform_prospeo_contact(r, search_params) for r in data]

        return {
            "success": True, "from_cache": True,
            "search_params": search_params,
            "leads": leads, "data": data,
            "count": len(leads), "total": len(leads),
            "created_at": cached.created_at.isoformat() if cached.created_at else None,
        }


@app.get("/debug")
async def debug_info():
    async with async_session() as session:
        all_cached = (await session.execute(select(CachedSearch))).scalars().all()
        searches = []
        for cached in all_cached:
            results = json.loads(cached.results)
            searches.append({
                "hash": cached.search_hash,
                "params": json.loads(cached.search_params),
                "result_count": len(results),
                "sample_lead": results[0] if results else None,
                "created_at": cached.created_at.isoformat() if cached.created_at else None,
            })
    return {"total_cached_searches": len(searches), "searches": searches}


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
