"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Seamless.ai API results to reduce costs.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
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
    seamless_api_key: str
    anthropic_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

    @property
    def async_database_url(self) -> str:
        """Ensure the database URL uses the asyncpg driver."""
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


settings = Settings()


# =============================================================================
# Seamless.ai Filter Mappings
# =============================================================================

# Seniority string -> Seamless.ai seniority label
SENIORITY_MAP = {
    "entry":    "Entry Level",
    "training": "Entry Level",
    "junior":   "Junior",
    "senior":   "Senior",
    "manager":  "Manager",
    "director": "Director",
    "partner":  "Director",
    "vp":       "VP",
    "c_suite":  "C-Suite",
    "cxo":      "C-Suite",
    "owner":    "Owner",
}

# Company size string -> Seamless.ai employee range label
COMPANY_SIZE_MAP = {
    "1-10":       "1-10",
    "11-50":      "11-50",
    "51-200":     "51-200",
    "201-500":    "201-500",
    "501-1000":   "501-1000",
    "1001-5000":  "1001-5000",
    "5001-10000": "5001-10000",
    "10001+":     "10001+",
}

# Valid Seamless.ai signal names
VALID_SIGNALS = {"promotion", "job_change", "all_signals"}

# Location type detection
_CONTINENTS = {"europe", "eu", "north america", "south america", "asia", "africa",
               "oceania", "middle east", "latam", "latin america", "apac", "asia pacific"}
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


def parse_location(location: str) -> dict:
    """Detect whether a location string is a city, state, country, or continent."""
    loc = location.strip()
    loc_lower = loc.lower()
    if loc_lower in _CONTINENTS:
        continent = "Europe" if loc_lower in ("europe", "eu") else loc
        return {"continent": continent}
    if loc_lower in _COUNTRIES:
        canonical = "United States" if loc_lower in ("us", "usa") else \
                    "United Kingdom" if loc_lower in ("uk",) else loc
        return {"country": canonical}
    if loc_lower in _US_STATES:
        return {"state": loc, "country": "United States"}
    return {"city": loc}


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
    """Create tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class SearchRequest(BaseModel):
    # Plain-text ICP query (auto-parsed if no structured filters provided)
    query: Optional[str] = None

    # Basic filters
    job_title: Optional[str] = None
    departments: Optional[list[str]] = None
    seniority: Optional[str] = None
    location: Optional[str] = None
    company: Optional[str] = None
    company_size: Optional[str] = None

    # Niche / industry filters
    industry: Optional[str] = None
    technologies: Optional[list[str]] = None
    keywords: Optional[str] = None

    # Buyer intent
    intent_topics: Optional[list[str]] = None

    # Revenue range
    revenue_min: Optional[int] = None
    revenue_max: Optional[int] = None

    # Career signals
    signals: Optional[list[str]] = None
    signals_since_days: int = 90

    limit: int = 10

    # Coerce string fields that Lovable may accidentally send as arrays
    @field_validator("keywords", "seniority", "job_title", "location",
                     "company", "industry", "company_size", "query", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return v

    # Coerce list fields that may be sent as comma-separated strings
    @field_validator("technologies", "departments", "intent_topics", "signals", mode="before")
    @classmethod
    def coerce_list(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


class ICPParseRequest(BaseModel):
    text: str  # Plain-English ICP description


async def load_industry_map():
    """No-op: Seamless.ai accepts industry as a plain string, no ID lookup needed."""
    pass


class SearchResponse(BaseModel):
    success: bool = True
    source: str  # "cache" or "api"
    from_cache: bool = False
    count: int
    total: int
    leads: list
    data: list


# =============================================================================
# Helper Functions
# =============================================================================

def transform_seamless_contact(contact: dict, search_params: dict = None) -> dict:
    """Transform a Seamless.ai contact object to the internal lead format."""
    search_params = search_params or {}

    full_name = f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip() or None

    return {
        "contact_name": full_name,
        "job_title": contact.get("title"),
        "company_name": contact.get("company_name"),
        "company_domain": contact.get("company_website"),
        "business_email": contact.get("email"),
        "linkedin_url": contact.get("linkedin_url"),
        "industry": search_params.get("industry"),
        "technologies": search_params.get("technologies"),
        "intent_topics": search_params.get("intent_topics"),
        "company_size": search_params.get("company_size"),
        "country": search_params.get("location"),
        "raw_data": contact,
    }


def generate_search_hash(params: dict) -> str:
    """Generate a consistent hash for search parameters."""
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_params.encode()).hexdigest()


async def fetch_from_seamless(params: dict) -> list:
    """
    Call Seamless.ai Contacts API to search for leads.

    Single-step workflow:
      POST /v1/contacts/search -> returns enriched contact profiles directly
    """
    searchable_fields = (
        "job_title", "departments", "location", "industry", "company",
        "company_size", "seniority", "technologies", "intent_topics",
        "keywords", "revenue_min", "revenue_max", "signals",
    )
    if not any(params.get(k) for k in searchable_fields):
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    headers = {
        "Authorization": f"Bearer {settings.seamless_api_key}",
        "Content-Type": "application/json",
    }

    # Build criteria array
    criteria: list[dict] = []

    if params.get("job_title"):
        criteria.append({"name": "job_title", "value": [params["job_title"]]})

    if params.get("departments"):
        criteria.append({"name": "department", "value": params["departments"]})

    if params.get("seniority"):
        level = SENIORITY_MAP.get(params["seniority"].lower())
        if level:
            criteria.append({"name": "seniority", "value": [level]})

    if params.get("location"):
        loc = parse_location(params["location"])
        if "city" in loc:
            criteria.append({"name": "city", "value": [loc["city"]]})
        elif "state" in loc:
            criteria.append({"name": "state", "value": [loc["state"]]})
        elif "country" in loc:
            criteria.append({"name": "country", "value": [loc["country"]]})
        elif "continent" in loc:
            criteria.append({"name": "continent", "value": [loc["continent"]]})

    if params.get("keywords"):
        criteria.append({"name": "keywords", "value": [params["keywords"]]})

    if params.get("company"):
        criteria.append({"name": "company_name", "value": [params["company"]]})

    if params.get("industry"):
        criteria.append({"name": "industry", "value": [params["industry"]]})

    if params.get("company_size"):
        size = COMPANY_SIZE_MAP.get(params["company_size"])
        if size:
            criteria.append({"name": "employee_count", "value": [size]})

    if params.get("technologies"):
        criteria.append({"name": "technologies", "value": params["technologies"]})

    if params.get("intent_topics"):
        criteria.append({"name": "intent_topics", "value": params["intent_topics"]})

    if params.get("revenue_min") or params.get("revenue_max"):
        revenue_filter: dict = {}
        if params.get("revenue_min"):
            revenue_filter["min"] = params["revenue_min"]
        if params.get("revenue_max"):
            revenue_filter["max"] = params["revenue_max"]
        criteria.append({"name": "revenue", "value": [revenue_filter]})

    if params.get("signals"):
        valid = [s for s in params["signals"] if s in VALID_SIGNALS]
        if valid:
            criteria.append({"name": "signals", "value": valid})

    search_body: dict = {
        "page": 1,
        "page_size": max(min(params.get("limit", 10), 100), 10),
        "criteria": criteria,
    }

    print(f"Seamless.ai request body: {json.dumps(search_body)}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = None
        for attempt in range(3):
            resp = await client.post(
                "https://api.seamless.ai/v1/contacts/search",
                headers=headers,
                json=search_body,
            )
            print(f"Seamless.ai search status: {resp.status_code}")
            if resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Seamless.ai 429 rate limit — retrying in {wait}s (attempt {attempt + 1}/3)")
            await asyncio.sleep(wait)

        print(f"Seamless.ai response body: {resp.text[:300]}")

        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Seamless.ai rate limit reached — please try again in a moment")
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Seamless.ai search error: {resp.text}",
            )

        return resp.json().get("contacts", [])


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches Seamless.ai API results to reduce costs",
    version="3.0.0"
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
    await load_industry_map()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# =============================================================================
# ICP Parser Endpoint
# =============================================================================

@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """
    Parse a plain-English ICP description into structured Seamless.ai search filters.

    Example input:
      "CTOs at fintech startups using Salesforce, 50-200 employees in NYC,
       showing intent to buy cybersecurity tools"

    Returns structured filters ready to pass directly into POST /search.
    """
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured on server")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""You are an ICP (Ideal Customer Profile) parser for a B2B lead generation platform.

Parse the following description into structured search filters. Return ONLY valid JSON with these fields (omit fields not mentioned or not clearly implied):

{{
  "job_title": "string — specific job title, e.g. CTO, VP of Sales",
  "departments": ["array of strings — e.g. engineering, sales, marketing, finance, product, hr, legal, operations, executive"],
  "seniority": "one of: entry, junior, senior, manager, director, vp, c_suite, owner",
  "location": "city name string, e.g. San Francisco",
  "company": "specific company name if mentioned",
  "company_size": "one of: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+",
  "industry": "industry keyword string, e.g. fintech, saas, healthcare, retail",
  "technologies": ["array of tech stack strings, e.g. Salesforce, HubSpot, AWS, Stripe"],
  "keywords": "additional free-text search terms",
  "intent_topics": ["array of strings describing what they want to buy or solve, e.g. cybersecurity, HR software, data analytics"],
  "revenue_min": integer in USD,
  "revenue_max": integer in USD,
  "signals": ["subset of: promotion, companyChange, allSignals — only if career signals are mentioned"]
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
    Cache-first lead search endpoint.

    Flow:
    1. Check local DB for cached results
    2. If found, return from cache
    3. If not found, fetch from Seamless.ai API
    4. Cache the raw results
    5. Return transformed results
    """
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != "" and v != []}

    print(f"=== SEARCH REQUEST ===")
    print(f"Filtered params: {params}")

    # If only a plain-text query was provided, auto-parse it into structured filters
    structured_fields = {
        "job_title", "departments", "seniority", "location", "company",
        "company_size", "industry", "technologies", "keywords",
        "intent_topics", "revenue_min", "revenue_max", "signals",
    }
    if params.get("query") and not any(params.get(f) for f in structured_fields):
        if not settings.anthropic_api_key:
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured — cannot parse plain-text query")
        print(f"Auto-parsing query via ICP parser: {params['query']}")
        parsed = await parse_icp(ICPParseRequest(text=params["query"]))
        parsed_filters = parsed.get("filters", {})
        for k, v in parsed_filters.items():
            if v is not None and v != "" and v != []:
                params[k] = v
        params.pop("query", None)
        print(f"Parsed into: {params}")

    search_hash = generate_search_hash(params)
    print(f"Search hash: {search_hash}")

    async with async_session() as session:
        # Step 1: Check local DB
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached:
            print(f"Cache HIT for hash: {search_hash}")
            data = json.loads(cached.results)
            leads = [transform_seamless_contact(lead, params) for lead in data]
            return SearchResponse(
                success=True,
                source="cache",
                from_cache=True,
                count=len(leads),
                total=len(leads),
                leads=leads,
                data=data,
            )

        # Step 3: Fetch from Seamless.ai API
        print(f"Cache MISS - calling Seamless.ai API with params: {params}")
        raw_leads = await fetch_from_seamless(params)
        print(f"Seamless.ai returned {len(raw_leads)} leads")

        leads = [transform_seamless_contact(lead, params) for lead in raw_leads]

        # Step 4: Cache the raw results
        new_cache = CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(params),
            results=json.dumps(raw_leads),
        )
        session.add(new_cache)
        await session.commit()

        return SearchResponse(
            success=True,
            source="api",
            from_cache=False,
            count=len(leads),
            total=len(leads),
            leads=leads,
            data=raw_leads,
        )


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@app.delete("/cache")
async def clear_cache():
    """Clear all cached searches."""
    async with async_session() as session:
        await session.execute(CachedSearch.__table__.delete())
        await session.commit()
    return {"message": "Cache cleared"}


@app.delete("/cache/empty")
async def clear_empty_cache():
    """Clear only cached searches with 0 results."""
    async with async_session() as session:
        stmt = select(CachedSearch)
        result = await session.execute(stmt)
        all_cached = result.scalars().all()

        deleted = 0
        for cached in all_cached:
            results = json.loads(cached.results)
            if len(results) == 0:
                await session.delete(cached)
                deleted += 1

        await session.commit()
    return {"message": f"Cleared {deleted} empty cached searches"}


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics and recent searches."""
    async with async_session() as session:
        from sqlalchemy import func, desc

        count_stmt = select(func.count()).select_from(CachedSearch)
        count_result = await session.execute(count_stmt)
        count = count_result.scalar()

        recent_stmt = select(CachedSearch).order_by(desc(CachedSearch.created_at)).limit(10)
        recent_result = await session.execute(recent_stmt)
        recent = recent_result.scalars().all()

        recent_searches = []
        for search in recent:
            params = json.loads(search.search_params)
            results = json.loads(search.results)
            recent_searches.append({
                "search_hash": search.search_hash,
                "params": params,
                "result_count": len(results),
                "created_at": search.created_at.isoformat() if search.created_at else None,
            })

    return {"cached_searches": count, "recent_searches": recent_searches}


@app.get("/cache/all")
async def get_all_cached_leads():
    """Retrieve all cached leads (fallback when Seamless.ai credits are exhausted)."""
    async with async_session() as session:
        stmt = select(CachedSearch).order_by(CachedSearch.created_at.desc())
        result = await session.execute(stmt)
        all_cached = result.scalars().all()

        all_leads = []
        seen_contacts = set()

        for cached in all_cached:
            data = json.loads(cached.results)
            search_params = json.loads(cached.search_params)
            for contact in data:
                contact_data = contact.get("data", {})
                full_name = f"{contact_data.get('firstName', '')} {contact_data.get('lastName', '')}".strip()
                company = contact_data.get("companyName", "")
                key = f"{full_name}-{company}"
                if key not in seen_contacts:
                    seen_contacts.add(key)
                    all_leads.append(transform_seamless_contact(contact, search_params))

        return {
            "success": True,
            "from_cache": True,
            "leads": all_leads,
            "count": len(all_leads),
            "total": len(all_leads),
            "message": "All cached leads retrieved",
        }


@app.get("/cache/search/{search_hash}")
async def get_cached_search(search_hash: str):
    """Retrieve a specific cached search by hash."""
    async with async_session() as session:
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached:
            raise HTTPException(status_code=404, detail="Cached search not found")

        data = json.loads(cached.results)
        search_params = json.loads(cached.search_params)
        leads = [transform_seamless_contact(lead, search_params) for lead in data]

        return {
            "success": True,
            "from_cache": True,
            "search_params": search_params,
            "leads": leads,
            "data": data,
            "count": len(leads),
            "total": len(leads),
            "created_at": cached.created_at.isoformat() if cached.created_at else None,
        }


@app.get("/debug")
async def debug_info():
    """Debug endpoint to see current cache state."""
    async with async_session() as session:
        stmt = select(CachedSearch)
        result = await session.execute(stmt)
        all_cached = result.scalars().all()

        searches = []
        for cached in all_cached:
            params = json.loads(cached.search_params)
            results = json.loads(cached.results)
            searches.append({
                "hash": cached.search_hash,
                "params": params,
                "result_count": len(results),
                "sample_lead": results[0] if results else None,
                "created_at": cached.created_at.isoformat() if cached.created_at else None,
            })

    return {"total_cached_searches": len(searches), "searches": searches}


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
