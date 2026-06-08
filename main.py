"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Wiza API results to reduce costs.
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
    wiza_api_key: str
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

WIZA_BASE = "https://wiza.co/api"


# =============================================================================
# Wiza Filter Mappings
# =============================================================================

# Our seniority keys -> Wiza job_title_level values
SENIORITY_MAP = {
    "entry":    "entry",
    "training": "entry",
    "intern":   "entry",
    "junior":   "junior",
    "senior":   "senior",
    "manager":  "manager",
    "head":     "manager",
    "director": "director",
    "partner":  "director",
    "vp":       "vp",
    "c_suite":  "c_suite",
    "cxo":      "c_suite",
    "owner":    "owner",
    "founder":  "owner",
}

# Our size keys -> Wiza company_size values (Wiza uses same range strings)
COMPANY_SIZE_MAP = {
    "1-10":       ["1-10"],
    "11-50":      ["11-50"],
    "51-200":     ["51-200"],
    "201-500":    ["201-500"],
    "501-1000":   ["501-1000"],
    "1001-5000":  ["1001-5000"],
    "5001-10000": ["5001-10000"],
    "10001+":     ["10001+"],
}

# Buyer intent -> Wiza funding_stage values
INTENT_TO_FUNDING = {
    "Funding":    ["seed", "series_a", "series_b", "series_c", "series_d",
                   "series_e", "series_f", "angel", "pre_seed"],
    "IPO":        ["ipo"],
    "Mergers":    ["private_equity", "post_ipo_equity"],
    "Investment": ["seed", "series_a", "series_b", "angel"],
}


def f(value: str, flag: str = "i") -> dict:
    """Wiza filter value wrapper — f='i' include, f='e' exclude."""
    return {"v": value, "f": flag}


# =============================================================================
# Database Models
# =============================================================================

class Base(DeclarativeBase):
    pass


class CachedSearch(Base):
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
    query: Optional[str] = None

    # Contact filters
    job_title: Optional[str] = None
    departments: Optional[list[str]] = None
    seniority: Optional[str] = None
    location: Optional[str] = None
    company_location: Optional[str] = None

    # Company filters
    company: Optional[str] = None
    company_domain: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    technologies: Optional[list[str]] = None
    keywords: Optional[str] = None

    # Buyer intent — maps to Wiza funding signals
    # Values: Funding, IPO, Mergers, Investment
    intent_topics: Optional[list[str]] = None

    # Revenue range (USD)
    revenue_min: Optional[int] = None
    revenue_max: Optional[int] = None

    # Career change signal
    job_change_days: Optional[int] = None

    # Legacy fields for backwards compatibility
    signals: Optional[list[str]] = None
    signals_since_days: int = 90
    intent_days: int = 90

    limit: int = 10

    @field_validator("keywords", "seniority", "job_title", "location", "company_location",
                     "company", "company_domain", "industry", "company_size", "query", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return v

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
    source: str
    from_cache: bool = False
    count: int
    total: int
    leads: list
    data: list


# =============================================================================
# Transform
# =============================================================================

def transform_wiza_contact(contact: dict, search_params: dict = None) -> dict:
    """Transform a Wiza contact object to the internal lead format."""
    search_params = search_params or {}

    emails = contact.get("emails") or []
    primary_email = (
        next((e["email"] for e in emails if e.get("type") == "work"), None)
        or (emails[0]["email"] if emails else None)
        or contact.get("email")
    )

    phones = contact.get("phones") or []
    primary_phone = (
        phones[0].get("pretty_number") or phones[0].get("number")
        if phones else contact.get("mobile_phone") or contact.get("phone_number")
    )

    return {
        "contact_name": contact.get("full_name"),
        "first_name": contact.get("first_name"),
        "last_name": contact.get("last_name"),
        "job_title": contact.get("title"),
        "linkedin_url": contact.get("linkedin"),
        "business_email": primary_email,
        "email_status": contact.get("email_status"),
        "phone": primary_phone,
        "location": contact.get("location"),
        "company_name": contact.get("name"),
        "company_domain": contact.get("domain"),
        "industry": contact.get("industry") or search_params.get("industry"),
        "company_size": contact.get("size") or search_params.get("company_size"),
        "company_revenue": contact.get("revenue"),
        "company_funding": contact.get("funding"),
        "technologies": search_params.get("technologies"),
        "raw_data": contact,
    }


def generate_search_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


# =============================================================================
# Wiza Workflow: Create List → Poll → Fetch Contacts
# =============================================================================

async def fetch_from_wiza(params: dict) -> list:
    """
    Wiza three-step workflow:
      1. POST /prospects/create_prospect_list  — create enrichment job with filters
      2. GET  /lists/{id}                      — poll until finished_at is set
      3. GET  /lists/{id}/contacts             — fetch enriched contacts
    """
    # Map legacy signals to job_change_days
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
        "Authorization": f"Bearer {settings.wiza_api_key}",
        "Content-Type": "application/json",
    }

    # ------------------------------------------------------------------
    # Build filters
    # ------------------------------------------------------------------
    filters: dict = {}

    if params.get("job_title"):
        filters["job_title"] = [f(params["job_title"])]

    if params.get("seniority"):
        level = SENIORITY_MAP.get(params["seniority"].lower())
        if level:
            filters["job_title_level"] = [level]

    if params.get("departments"):
        filters["job_role"] = [f(d) for d in params["departments"]]

    if params.get("location"):
        filters["location"] = [f(params["location"])]

    if params.get("company_location"):
        filters["company_location"] = [f(params["company_location"])]

    if params.get("company"):
        filters["job_company"] = [f(params["company"])]

    if params.get("industry"):
        filters["company_industry"] = [f(params["industry"])]

    if params.get("company_size"):
        sizes = COMPANY_SIZE_MAP.get(params["company_size"], [])
        if sizes:
            filters["company_size"] = sizes

    if params.get("technologies"):
        filters["skill"] = [f(t) for t in params["technologies"]]

    if params.get("keywords"):
        # Wiza doesn't have a generic keyword field — apply to job title search
        existing = filters.get("job_title", [])
        filters["job_title"] = existing + [f(params["keywords"])]

    # Buyer intent via funding signals
    if params.get("intent_topics"):
        funding_stages = []
        for topic in params["intent_topics"]:
            funding_stages.extend(INTENT_TO_FUNDING.get(topic, []))
        if funding_stages:
            filters["funding_stage"] = list(set(funding_stages))

    # Revenue range
    if params.get("revenue_min"):
        filters["revenue"] = filters.get("revenue", {})
        filters["revenue"]["min"] = params["revenue_min"]
    if params.get("revenue_max"):
        filters["revenue"] = filters.get("revenue", {})
        filters["revenue"]["max"] = params["revenue_max"]

    limit = max(min(params.get("limit", 10), 100), 1)
    list_name = f"salesos-{generate_search_hash(params)[:8]}-{int(datetime.utcnow().timestamp())}"

    body = {
        "list": {
            "name": list_name,
            "max_profiles": limit,
            "enrichment_level": "partial",
            "email_options": {
                "accept_work": True,
                "accept_personal": False,
                "accept_generic": False,
            },
            "skip_duplicates": True,
        },
        "filters": filters,
    }

    print(f"Wiza create_prospect_list body: {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ---- Step 1: Create prospect list ----
        create_resp = None
        for attempt in range(3):
            create_resp = await client.post(
                f"{WIZA_BASE}/prospects/create_prospect_list",
                headers=headers,
                json=body,
            )
            print(f"Wiza create status: {create_resp.status_code}")
            if create_resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Rate limited — retrying in {wait}s")
            await asyncio.sleep(wait)

        print(f"Wiza create response: {create_resp.text[:400]}")

        if create_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Wiza rate limit reached — please try again in a moment")
        if create_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=create_resp.status_code,
                detail=f"Wiza list creation error: {create_resp.text}",
            )

        list_id = create_resp.json().get("data", {}).get("id")
        if not list_id:
            print("Wiza returned no list ID")
            return []

        print(f"Wiza list created: id={list_id}")

        # ---- Step 2: Poll until finished ----
        max_polls = 30
        poll_delay = 5.0

        for poll_num in range(max_polls):
            await asyncio.sleep(poll_delay)

            poll_resp = await client.get(
                f"{WIZA_BASE}/lists/{list_id}",
                headers=headers,
            )
            print(f"Wiza poll #{poll_num + 1} status: {poll_resp.status_code}")

            if poll_resp.status_code not in (200, 201):
                print(f"Wiza poll error: {poll_resp.text[:200]}")
                break

            list_data = poll_resp.json().get("data", {})
            finished_at = list_data.get("finished_at")
            list_status = list_data.get("status", "")

            print(f"Wiza list status: {list_status}, finished_at: {finished_at}")

            if finished_at or list_status in ("complete", "finished", "done"):
                print(f"Wiza list finished after {poll_num + 1} poll(s)")
                break

            if list_status in ("failed", "error"):
                print(f"Wiza list failed: {list_data}")
                return []

            poll_delay = min(poll_delay * 1.2, 15.0)

        # ---- Step 3: Fetch contacts ----
        contacts_resp = await client.get(
            f"{WIZA_BASE}/lists/{list_id}/contacts",
            headers=headers,
            params={"segment": "valid"},
        )
        print(f"Wiza contacts status: {contacts_resp.status_code}")
        print(f"Wiza contacts response: {contacts_resp.text[:400]}")

        if contacts_resp.status_code not in (200, 201):
            # Try fetching all contacts if "valid" segment returns nothing
            contacts_resp = await client.get(
                f"{WIZA_BASE}/lists/{list_id}/contacts",
                headers=headers,
                params={"segment": "people"},
            )

        if contacts_resp.status_code not in (200, 201):
            return []

        contacts = contacts_resp.json().get("data", [])
        print(f"Wiza returned {len(contacts)} contacts")
        return contacts


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches Wiza API results to reduce costs",
    version="6.0.0"
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
        print("App will continue — DB retried on first request.")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# =============================================================================
# ICP Parser Endpoint
# =============================================================================

@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """Parse a plain-English ICP description into structured Wiza search filters."""
    if not settings.anthropic_api_key:
        return {"success": True, "filters": {"keywords": request.text}}

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""You are an ICP (Ideal Customer Profile) parser for a B2B lead generation platform powered by Wiza.

Parse the following description into structured search filters. Return ONLY valid JSON (omit fields not mentioned):

{{
  "job_title": "string — e.g. CTO, VP of Sales",
  "departments": ["array — e.g. Engineering, Sales, Marketing, Finance, Product, HR, Legal, Operations"],
  "seniority": "one of: entry, junior, senior, manager, director, vp, c_suite, owner, founder",
  "location": "city, state, or country — e.g. New York, California, United States",
  "company_location": "company HQ location",
  "company": "specific company name if mentioned",
  "company_domain": "company domain e.g. acme.com",
  "company_size": "one of: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+",
  "industry": "industry string e.g. Software Development, Financial Services, Healthcare",
  "technologies": ["tech stack strings e.g. Salesforce, HubSpot, AWS"],
  "keywords": "additional search terms",
  "intent_topics": ["buyer intent — only from: Funding, IPO, Mergers, Investment"],
  "revenue_min": integer in USD,
  "revenue_max": integer in USD,
  "job_change_days": integer e.g. 90 if recently changed jobs is mentioned
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
    Cache-first lead search powered by Wiza.

    Flow:
    1. Check DB cache — return immediately if found
    2. Cache miss → Wiza 3-step: create list → poll → fetch contacts
    3. Cache and return results
    """
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != "" and v != []}
    print(f"=== SEARCH REQUEST ===\nFiltered params: {params}")

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
                print(f"Auto-parsing query: {raw_query}")
                icp_result = await parse_icp(ICPParseRequest(text=raw_query))
                parsed_filters = icp_result.get("filters", {})
                print(f"ICP parser returned: {parsed_filters}")
            except Exception as e:
                print(f"WARNING: ICP parser failed ({e}) — falling back to keyword search")

        for k, v in parsed_filters.items():
            if v is not None and v != "" and v != []:
                params[k] = v

        if not any(params.get(f) for f in structured_fields):
            print("ICP parse yielded no filters — using raw query as keyword search")
            params["keywords"] = raw_query

        print(f"Final params: {params}")

    search_hash = generate_search_hash(params)
    print(f"Search hash: {search_hash}")

    async with async_session() as session:
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        cached = (await session.execute(stmt)).scalar_one_or_none()

        if cached:
            print(f"Cache HIT for hash: {search_hash}")
            data = json.loads(cached.results)
            leads = [transform_wiza_contact(r, params) for r in data]
            return SearchResponse(
                success=True, source="cache", from_cache=True,
                count=len(leads), total=len(leads), leads=leads, data=data,
            )

        print(f"Cache MISS — calling Wiza")
        raw_results = await fetch_from_wiza(params)
        print(f"Wiza returned {len(raw_results)} leads")

        leads = [transform_wiza_contact(r, params) for r in raw_results]

        session.add(CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(params),
            results=json.dumps(raw_results),
        ))
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
        all_cached = (await session.execute(select(CachedSearch))).scalars().all()
        deleted = sum(1 for c in all_cached if len(json.loads(c.results)) == 0)
        for c in all_cached:
            if len(json.loads(c.results)) == 0:
                await session.delete(c)
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
    """Retrieve all cached leads (fallback when Wiza credits are exhausted)."""
    async with async_session() as session:
        all_cached = (await session.execute(
            select(CachedSearch).order_by(CachedSearch.created_at.desc())
        )).scalars().all()

        all_leads, seen = [], set()
        for cached in all_cached:
            data = json.loads(cached.results)
            search_params = json.loads(cached.search_params)
            for contact in data:
                key = f"{contact.get('full_name', '')}-{contact.get('name', '')}"
                if key not in seen:
                    seen.add(key)
                    all_leads.append(transform_wiza_contact(contact, search_params))

        return {
            "success": True, "from_cache": True,
            "leads": all_leads, "count": len(all_leads), "total": len(all_leads),
            "message": "All cached leads retrieved",
        }


@app.get("/cache/search/{search_hash}")
async def get_cached_search(search_hash: str):
    async with async_session() as session:
        cached = (await session.execute(
            select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        )).scalar_one_or_none()
        if not cached:
            raise HTTPException(status_code=404, detail="Cached search not found")

        data = json.loads(cached.results)
        search_params = json.loads(cached.search_params)
        leads = [transform_wiza_contact(r, search_params) for r in data]
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
        return {
            "total_cached_searches": len(all_cached),
            "searches": [
                {
                    "hash": c.search_hash,
                    "params": json.loads(c.search_params),
                    "result_count": len(json.loads(c.results)),
                    "sample_lead": json.loads(c.results)[0] if json.loads(c.results) else None,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in all_cached
            ],
        }


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
