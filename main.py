"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Lusha API results to reduce costs.
"""

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
    lusha_api_key: str
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
# Lusha Filter Mappings
# =============================================================================

# Seniority string -> Lusha numeric string
SENIORITY_MAP = {
    "entry":    "1",
    "training": "1",
    "junior":   "2",
    "senior":   "3",
    "manager":  "4",
    "director": "5",
    "partner":  "5",
    "vp":       "6",
    "c_suite":  "7",
    "cxo":      "7",
    "owner":    "7",
}

# Company size string -> Lusha sizes object
COMPANY_SIZE_MAP = {
    "1-10":       {"min": 1,     "max": 10},
    "11-50":      {"min": 11,    "max": 50},
    "51-200":     {"min": 51,    "max": 200},
    "201-500":    {"min": 201,   "max": 500},
    "501-1000":   {"min": 501,   "max": 1000},
    "1001-5000":  {"min": 1001,  "max": 5000},
    "5001-10000": {"min": 5001,  "max": 10000},
    "10001+":     {"min": 10001, "max": None},
}

# Valid Lusha signal names
VALID_SIGNALS = {"promotion", "companyChange", "allSignals"}

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

def transform_lusha_contact(contact: dict, search_params: dict = None) -> dict:
    """Transform a Lusha enriched contact object to the internal lead format."""
    data = contact.get("data", {})
    search_params = search_params or {}

    full_name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip() or None

    # Prefer work email; fall back to first available address
    emails = data.get("emailAddresses", [])
    work_email = next(
        (e["email"] for e in emails if e.get("emailType") == "work"),
        emails[0]["email"] if emails else None,
    )

    # Extract LinkedIn URL from socialLinks array
    social_links = data.get("socialLinks", [])
    linkedin_url = next(
        (s["url"] for s in social_links if "linkedin" in s.get("url", "").lower()),
        None,
    )

    return {
        "contact_name": full_name,
        "job_title": data.get("jobTitle"),
        "company_name": data.get("companyName"),
        "company_domain": data.get("companyWebsite"),
        "business_email": work_email,
        "linkedin_url": linkedin_url,
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


async def fetch_from_lusha(params: dict) -> list:
    """
    Call Lusha Prospecting API to search for and enrich leads.

    Two-step workflow:
      1. POST /prospecting/contact/search  -> returns contact IDs + requestId
      2. POST /prospecting/contact/enrich  -> returns full profiles for those IDs
    """
    searchable_fields = (
        "job_title", "departments", "location", "industry", "company",
        "company_size", "seniority", "technologies", "intent_topics",
        "keywords", "revenue_min", "revenue_max", "signals",
    )
    if not any(params.get(k) for k in searchable_fields):
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    headers = {
        "api_key": settings.lusha_api_key,
        "Content-Type": "application/json",
    }

    # --- Contact-level filters ---
    contact_include: dict = {
        "existing_data_points": ["work_email"],
    }

    if params.get("job_title"):
        contact_include["jobTitles"] = [params["job_title"]]

    if params.get("departments"):
        contact_include["departments"] = params["departments"]

    if params.get("seniority"):
        lusha_level = SENIORITY_MAP.get(params["seniority"].lower())
        if lusha_level:
            contact_include["seniority"] = [lusha_level]

    if params.get("location"):
        contact_include["locations"] = [parse_location(params["location"])]

    if params.get("keywords"):
        contact_include["searchText"] = params["keywords"]

    if params.get("signals"):
        valid = [s for s in params["signals"] if s in VALID_SIGNALS]
        if valid:
            since_days = params.get("signals_since_days", 90)
            start_date = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")
            contact_include["signals"] = {"names": valid, "startDate": start_date}

    # --- Company-level filters ---
    company_include: dict = {}

    if params.get("company"):
        company_include["names"] = [params["company"]]

    if params.get("industry"):
        # Use searchText for flexible industry matching (IDs required for mainIndustriesIds)
        company_include["searchText"] = params["industry"]

    if params.get("company_size"):
        size_range = COMPANY_SIZE_MAP.get(params["company_size"])
        if size_range:
            company_include["sizes"] = [size_range]

    if params.get("technologies"):
        company_include["technologies"] = params["technologies"]

    if params.get("intent_topics"):
        company_include["intentTopics"] = params["intent_topics"]

    if params.get("revenue_min") or params.get("revenue_max"):
        revenue = {}
        if params.get("revenue_min"):
            revenue["min"] = params["revenue_min"]
        if params.get("revenue_max"):
            revenue["max"] = params["revenue_max"]
        company_include["revenues"] = [revenue]

    search_body: dict = {
        "pages": {"page": 0, "size": max(min(params.get("limit", 10), 40), 10)},
        "filters": {"contacts": {"include": contact_include}},
    }
    if company_include:
        search_body["filters"]["companies"] = {"include": company_include}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Search — get matching contact IDs
        search_resp = await client.post(
            "https://api.lusha.com/prospecting/contact/search",
            headers=headers,
            json=search_body,
        )

        if search_resp.status_code == 404:
            return []
        if search_resp.status_code != 200:
            raise HTTPException(
                status_code=search_resp.status_code,
                detail=f"Lusha search error: {search_resp.text}",
            )

        search_data = search_resp.json()
        request_id = search_data.get("requestId")
        contacts = search_data.get("data", [])

        if not contacts or not request_id:
            return []

        # Step 2: Enrich — get full profiles for the matched contact IDs
        enrich_resp = await client.post(
            "https://api.lusha.com/prospecting/contact/enrich",
            headers=headers,
            json={
                "requestId": request_id,
                "contactIds": [c["contactId"] for c in contacts],
            },
        )

        if enrich_resp.status_code != 200:
            raise HTTPException(
                status_code=enrich_resp.status_code,
                detail=f"Lusha enrich error: {enrich_resp.text}",
            )

        return enrich_resp.json().get("contacts", [])


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches Lusha API results to reduce costs",
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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# =============================================================================
# ICP Parser Endpoint
# =============================================================================

@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """
    Parse a plain-English ICP description into structured Lusha search filters.

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
    3. If not found, fetch from Lusha API (search + enrich)
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
            leads = [transform_lusha_contact(lead, params) for lead in data]
            return SearchResponse(
                success=True,
                source="cache",
                from_cache=True,
                count=len(leads),
                total=len(leads),
                leads=leads,
                data=data,
            )

        # Step 3: Fetch from Lusha API
        print(f"Cache MISS - calling Lusha API with params: {params}")
        raw_leads = await fetch_from_lusha(params)
        print(f"Lusha returned {len(raw_leads)} leads")

        leads = [transform_lusha_contact(lead, params) for lead in raw_leads]

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
    """Retrieve all cached leads (fallback when Lusha credits are exhausted)."""
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
                    all_leads.append(transform_lusha_contact(contact, search_params))

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
        leads = [transform_lusha_contact(lead, search_params) for lead in data]

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
