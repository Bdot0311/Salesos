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

SEAMLESS_BASE = "https://api.seamless.ai/api/client/v1"


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

# Company size string -> Seamless.ai companySize label
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

# Revenue range string -> Seamless.ai companyRevenue label
REVENUE_MAP = {
    (None,      1_000_000):   "$0 - $1M",
    (1_000_000, 10_000_000):  "$1M - $10M",
    (10_000_000, 50_000_000): "$10M - $50M",
    (50_000_000, 100_000_000):"$50M - $100M",
    (100_000_000, 250_000_000):"$100M - $250M",
    (250_000_000, 500_000_000):"$250M - $500M",
    (500_000_000, 1_000_000_000):"$500M - $1B",
    (1_000_000_000, None):    "$1B+",
}

# Buyer intent news/event types supported by Seamless.ai search
VALID_NEWS_TYPES = {
    "Acquisition", "IPO", "Expansion", "Leadership Change",
    "New Offering", "Partnership", "Investment", "Merger",
}

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
    if loc_lower in _COUNTRIES:
        canonical = "United States" if loc_lower in ("us", "usa") else \
                    "United Kingdom" if loc_lower in ("uk",) else loc
        return {"country": canonical}
    if loc_lower in _US_STATES:
        return {"state": loc.title()}
    return {"city": loc}


def resolve_revenue_label(revenue_min: int | None, revenue_max: int | None) -> list[str]:
    """Map revenue min/max to Seamless.ai revenue range label(s)."""
    matched = []
    for (rmin, rmax), label in REVENUE_MAP.items():
        lo = revenue_min or 0
        hi = revenue_max or float("inf")
        band_lo = rmin or 0
        band_hi = rmax or float("inf")
        if lo <= band_hi and hi >= band_lo:
            matched.append(label)
    return matched


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

    # Buyer intent — news/event types that signal purchase readiness
    # Valid values: Acquisition, IPO, Expansion, Leadership Change,
    #               New Offering, Partnership, Investment, Merger
    intent_topics: Optional[list[str]] = None
    # How far back to look for those events (60, 90, 180, or 365 days)
    intent_days: Optional[str] = "90"

    # Revenue range
    revenue_min: Optional[int] = None
    revenue_max: Optional[int] = None

    # Career signals (job_change / promotion)
    signals: Optional[list[str]] = None
    signals_since_days: int = 90

    limit: int = 10

    # Coerce string fields that Lovable may accidentally send as arrays
    @field_validator("keywords", "seniority", "job_title", "location",
                     "company", "industry", "company_size", "query",
                     "intent_days", mode="before")
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
    text: str


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
    """Transform a Seamless.ai enriched contact object to the internal lead format."""
    search_params = search_params or {}

    # Poll results nest the contact under a 'contact' key; search results are flat
    c = contact.get("contact", contact)

    name = c.get("name", "").strip() or \
           f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() or None

    # Prefer work email from enriched result; fall back to any email field
    emails = c.get("emailAddresses") or []
    if emails:
        work_email = next(
            (e.get("email") or e if isinstance(e, str) else None
             for e in emails if (e.get("type") == "work" if isinstance(e, dict) else True)),
            emails[0].get("email") if isinstance(emails[0], dict) else emails[0],
        )
    else:
        work_email = c.get("email")

    intent_signals = []
    for event in (c.get("newsAndEvents") or []):
        if isinstance(event, dict) and event.get("type"):
            intent_signals.append({"type": event["type"], "title": event.get("title"),
                                   "date": event.get("date"), "url": event.get("url")})

    return {
        "contact_name": name,
        "job_title": c.get("title"),
        "company_name": c.get("company"),
        "company_domain": c.get("domain"),
        "business_email": work_email,
        "phone": c.get("phone") or c.get("phoneNumbers", [None])[0] if c.get("phoneNumbers") else c.get("phone"),
        "linkedin_url": c.get("liUrl") or c.get("linkedinUrl"),
        "industry": (c.get("industries") or [None])[0] or search_params.get("industry"),
        "technologies": search_params.get("technologies"),
        "intent_signals": intent_signals or None,
        "job_change_alert": c.get("jobChangeAlert"),
        "company_size": c.get("employeeSizeRange") or search_params.get("company_size"),
        "company_revenue": c.get("companyRevenue"),
        "country": c.get("country") or search_params.get("location"),
        "city": c.get("city"),
        "state": c.get("state"),
        "seniority": c.get("seniority"),
        "department": c.get("department"),
        "raw_data": contact,
    }


def generate_search_hash(params: dict) -> str:
    """Generate a consistent hash for search parameters."""
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_params.encode()).hexdigest()


# =============================================================================
# Seamless.ai Three-Step Workflow
# =============================================================================

async def fetch_from_seamless(params: dict) -> list:
    """
    Three-step Seamless.ai workflow:
      1. POST /search/contacts    — find contacts with buyer intent filters
      2. POST /contacts/research  — enrich those contacts (async job)
      3. GET  /contacts/research/poll — poll until done, return full profiles
    """
    searchable_fields = (
        "job_title", "departments", "location", "industry", "company",
        "company_size", "seniority", "technologies", "intent_topics",
        "keywords", "revenue_min", "revenue_max", "signals",
    )
    if not any(params.get(k) for k in searchable_fields):
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    headers = {
        "Token": settings.seamless_api_key,
        "Content-Type": "application/json",
    }

    # ------------------------------------------------------------------
    # Step 1: Build search body
    # ------------------------------------------------------------------
    search_body: dict = {
        "limit": max(min(params.get("limit", 10), 100), 10),
    }

    if params.get("job_title"):
        search_body["jobTitle"] = [params["job_title"]]

    if params.get("departments"):
        search_body["department"] = params["departments"]

    if params.get("seniority"):
        level = SENIORITY_MAP.get(params["seniority"].lower())
        if level:
            search_body["seniority"] = [level]

    if params.get("location"):
        loc = parse_location(params["location"])
        if "city" in loc:
            search_body["contactCity"] = [loc["city"]]
            search_body["locationType"] = "contact"
        elif "state" in loc:
            search_body["contactState"] = [loc["state"]]
            search_body["locationType"] = "contact"
        elif "country" in loc:
            search_body["contactCountry"] = [loc["country"]]
            search_body["locationType"] = "contact"

    if params.get("keywords"):
        search_body["contactKeyword"] = [params["keywords"]]

    if params.get("company"):
        search_body["companyName"] = [params["company"]]

    if params.get("industry"):
        search_body["industry"] = [params["industry"]]

    if params.get("company_size"):
        size = COMPANY_SIZE_MAP.get(params["company_size"])
        if size:
            search_body["companySize"] = [size]

    if params.get("technologies"):
        search_body["technologies"] = params["technologies"]
        search_body["technologiesIsOr"] = True

    # Buyer intent: map intent_topics to newsTypes
    if params.get("intent_topics"):
        valid_news = [t for t in params["intent_topics"] if t in VALID_NEWS_TYPES]
        if valid_news:
            search_body["newsTypes"] = valid_news
            # intent_days must be one of "60", "90", "180", "365"
            days = str(params.get("intent_days", "90"))
            if days not in ("60", "90", "180", "365"):
                days = "90"
            search_body["newsTypeDates"] = [days]

    # Career signals
    if params.get("signals"):
        since_days = params.get("signals_since_days", 90)
        for sig in params["signals"]:
            if sig in ("promotion", "job_change"):
                search_body["jobChanges"] = {
                    "changeType": "New Promotion" if sig == "promotion" else "New Hire",
                    "dayRange": since_days,
                }
                break

    if params.get("revenue_min") or params.get("revenue_max"):
        labels = resolve_revenue_label(params.get("revenue_min"), params.get("revenue_max"))
        if labels:
            search_body["companyRevenue"] = labels

    print(f"Seamless.ai search body: {json.dumps(search_body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ---- Step 1: Search ----
        search_resp = None
        for attempt in range(3):
            search_resp = await client.post(
                f"{SEAMLESS_BASE}/search/contacts",
                headers=headers,
                json=search_body,
            )
            print(f"Seamless.ai search status: {search_resp.status_code}")
            if search_resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Rate limited — retrying in {wait}s (attempt {attempt + 1}/3)")
            await asyncio.sleep(wait)

        print(f"Seamless.ai search body (response): {search_resp.text[:400]}")

        if search_resp.status_code == 404:
            return []
        if search_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Seamless.ai rate limit reached — please try again in a moment")
        if search_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=search_resp.status_code,
                detail=f"Seamless.ai search error: {search_resp.text}",
            )

        search_data = search_resp.json()
        contacts = search_data.get("data", [])
        print(f"Seamless.ai search returned {len(contacts)} contacts")

        if not contacts:
            return []

        search_result_ids = [c["searchResultId"] for c in contacts if c.get("searchResultId")]
        if not search_result_ids:
            # Search-only results have no emails — return what we have
            return contacts

        # ---- Step 2: Research (enrich) ----
        research_resp = None
        for attempt in range(3):
            research_resp = await client.post(
                f"{SEAMLESS_BASE}/contacts/research",
                headers=headers,
                json={"searchResultIds": search_result_ids},
            )
            print(f"Seamless.ai research status: {research_resp.status_code}")
            if research_resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Rate limited — retrying in {wait}s (attempt {attempt + 1}/3)")
            await asyncio.sleep(wait)

        print(f"Seamless.ai research response: {research_resp.text[:400]}")

        if research_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Seamless.ai rate limit reached — please try again in a moment")
        if research_resp.status_code not in (200, 201, 202):
            raise HTTPException(
                status_code=research_resp.status_code,
                detail=f"Seamless.ai research error: {research_resp.text}",
            )

        request_ids: list[str] = research_resp.json().get("requestIds", [])
        if not request_ids:
            return contacts  # fall back to search-only data

        # ---- Step 3: Poll until all requests are done ----
        enriched: list[dict] = []
        max_polls = 20
        poll_delay = 3.0

        for poll_num in range(max_polls):
            await asyncio.sleep(poll_delay)

            poll_resp = await client.get(
                f"{SEAMLESS_BASE}/contacts/research/poll",
                headers=headers,
                params={"requestIds": ",".join(request_ids)},
            )
            print(f"Poll #{poll_num + 1} status: {poll_resp.status_code}")

            if poll_resp.status_code not in (200, 201):
                print(f"Poll error: {poll_resp.text[:200]}")
                break

            poll_data = poll_resp.json().get("data", [])
            done_statuses = {"done", "error", "missing", "duplicate",
                             "not found", "contact-already-researched"}

            all_done = all(r.get("status", "").lower() in done_statuses for r in poll_data)

            for result in poll_data:
                if result.get("status", "").lower() == "done" and result.get("contact"):
                    enriched.append(result)

            if all_done:
                print(f"All research requests done after {poll_num + 1} poll(s)")
                break

            poll_delay = min(poll_delay * 1.5, 10.0)

        if enriched:
            return enriched

        # Fell through polling — return search-only contacts as fallback
        print("WARNING: Enrichment timed out or returned no results — returning search data")
        return contacts


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches Seamless.ai API results to reduce costs",
    version="4.0.0"
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
    Parse a plain-English ICP description into structured Seamless.ai search filters.

    Example input:
      "CTOs at fintech startups using Salesforce, 50-200 employees in NYC,
       companies that recently had leadership changes or new funding"

    Returns structured filters ready to pass directly into POST /search.
    """
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured on server")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""You are an ICP (Ideal Customer Profile) parser for a B2B lead generation platform powered by Seamless.ai.

Parse the following description into structured search filters. Return ONLY valid JSON with these fields (omit fields not mentioned or not clearly implied):

{{
  "job_title": "string — specific job title, e.g. CTO, VP of Sales",
  "departments": ["array of strings — e.g. Engineering, Sales, Marketing, Finance, Product, HR, Legal, Operations, Executive"],
  "seniority": "one of: entry, junior, senior, manager, director, vp, c_suite, owner",
  "location": "city, US state, or country name string, e.g. San Francisco",
  "company": "specific company name if mentioned",
  "company_size": "one of: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+",
  "industry": "industry keyword string, e.g. fintech, SaaS, healthcare, retail",
  "technologies": ["array of tech stack strings, e.g. Salesforce, HubSpot, AWS, Stripe"],
  "keywords": "additional free-text search terms",
  "intent_topics": ["buyer intent news/event types — only choose from: Acquisition, IPO, Expansion, Leadership Change, New Offering, Partnership, Investment, Merger"],
  "intent_days": "one of: 60, 90, 180, 365 — lookback window for intent events",
  "revenue_min": integer in USD,
  "revenue_max": integer in USD,
  "signals": ["subset of: job_change, promotion — only if career signals are mentioned"]
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
    1. Check local DB for cached results
    2. If found, return from cache
    3. If not found, run Seamless.ai 3-step workflow:
       a. Search contacts (with intent filters)
       b. Enrich via research endpoint
       c. Poll for enriched profiles
    4. Cache and return results
    """
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != "" and v != []}

    print(f"=== SEARCH REQUEST ===")
    print(f"Filtered params: {params}")

    # Auto-parse plain-text query into structured filters
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
        # Check cache
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

        # Cache miss — call Seamless.ai
        print(f"Cache MISS — calling Seamless.ai with params: {params}")
        raw_leads = await fetch_from_seamless(params)
        print(f"Seamless.ai returned {len(raw_leads)} leads")

        leads = [transform_seamless_contact(lead, params) for lead in raw_leads]

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
            p = json.loads(search.search_params)
            results = json.loads(search.results)
            recent_searches.append({
                "search_hash": search.search_hash,
                "params": p,
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
                c = contact.get("contact", contact)
                name = c.get("name", "") or f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
                company = c.get("company", "")
                key = f"{name}-{company}"
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
            p = json.loads(cached.search_params)
            results = json.loads(cached.results)
            searches.append({
                "hash": cached.search_hash,
                "params": p,
                "result_count": len(results),
                "sample_lead": results[0] if results else None,
                "created_at": cached.created_at.isoformat() if cached.created_at else None,
            })

    return {"total_cached_searches": len(searches), "searches": searches}


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
