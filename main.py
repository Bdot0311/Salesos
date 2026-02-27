"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Lusha API results to reduce costs.
"""

import hashlib
import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# PDL seniority string -> Lusha numeric string
PDL_TO_LUSHA_SENIORITY = {
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

# PDL company size string -> Lusha employeesRange object
PDL_TO_LUSHA_COMPANY_SIZE = {
    "1-10":       {"min": 1,     "max": 10},
    "11-50":      {"min": 11,    "max": 50},
    "51-200":     {"min": 51,    "max": 200},
    "201-500":    {"min": 201,   "max": 500},
    "501-1000":   {"min": 501,   "max": 1000},
    "1001-5000":  {"min": 1001,  "max": 5000},
    "5001-10000": {"min": 5001,  "max": 10000},
    "10001+":     {"min": 10001, "max": None},
}


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
    job_title: Optional[str] = None   # e.g., "software engineer", "ceo", "data scientist"
    location: Optional[str] = None    # City name, e.g., "san francisco", "new york"
    industry: Optional[str] = None    # e.g., "computer software", "financial services"
    company: Optional[str] = None     # Company name, e.g., "google", "microsoft"
    company_size: Optional[str] = None  # "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"
    seniority: Optional[str] = None   # "entry", "junior", "senior", "manager", "director", "vp", "c_suite", "owner"
    limit: int = 10


class SearchResponse(BaseModel):
    success: bool = True
    source: str  # "cache" or "api"
    from_cache: bool = False
    count: int
    total: int
    leads: list  # Transformed lead format for frontend
    data: list   # Raw Lusha data for reference


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
        # Lusha returns these as input filters, not response fields — carry forward from search params
        "industry": search_params.get("industry"),
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
    if not any(params.get(k) for k in ("job_title", "location", "industry", "company", "company_size", "seniority")):
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    headers = {
        "api_key": settings.lusha_api_key,
        "Content-Type": "application/json",
    }

    # --- Contact-level filters ---
    contact_include: dict = {
        "existing_data_points": ["work_email"],  # only return contacts that have a work email
    }

    if params.get("job_title"):
        contact_include["jobTitles"] = [params["job_title"]]

    if params.get("seniority"):
        lusha_level = PDL_TO_LUSHA_SENIORITY.get(params["seniority"].lower())
        if lusha_level:
            contact_include["seniority"] = [lusha_level]

    if params.get("location"):
        contact_include["locations"] = [{"city": params["location"]}]

    # --- Company-level filters ---
    company_include: dict = {}

    if params.get("company"):
        company_include["name"] = [params["company"]]

    if params.get("industry"):
        company_include["mainIndustries"] = [params["industry"]]

    if params.get("company_size"):
        size_range = PDL_TO_LUSHA_COMPANY_SIZE.get(params["company_size"])
        if size_range:
            company_include["employeesRange"] = size_range

    search_body: dict = {
        "pages": {"page": 0, "size": max(min(params.get("limit", 10), 40), 10)},  # Lusha min=10, max=40
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
    version="2.0.0"
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
    """Health check endpoint."""
    return {"status": "healthy"}


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
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != ""}

    print(f"=== SEARCH REQUEST ===")
    print(f"Raw request: {request.model_dump()}")
    print(f"Filtered params: {params}")

    search_hash = generate_search_hash(params)
    print(f"Search hash: {search_hash}")

    async with async_session() as session:
        # Step 1: Check local DB
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached:
            # Step 2: Return from cache
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
                data=data
            )

        # Step 3: Fetch from Lusha API
        print(f"Cache MISS - calling Lusha API with params: {params}")
        raw_leads = await fetch_from_lusha(params)
        print(f"Lusha returned {len(raw_leads)} leads")

        # Transform leads to expected format
        leads = [transform_lusha_contact(lead, params) for lead in raw_leads]

        # Step 4: Cache the raw Lusha results
        new_cache = CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(params),
            results=json.dumps(raw_leads)
        )
        session.add(new_cache)
        await session.commit()

        # Step 5: Return new results
        return SearchResponse(
            success=True,
            source="api",
            from_cache=False,
            count=len(leads),
            total=len(leads),
            leads=leads,
            data=raw_leads
        )


@app.delete("/cache")
async def clear_cache():
    """Clear all cached searches (admin endpoint)."""
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


@app.get("/debug")
async def debug_info():
    """Debug endpoint to see current state."""
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
                "created_at": cached.created_at.isoformat() if cached.created_at else None
            })

    return {
        "total_cached_searches": len(searches),
        "searches": searches
    }


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
                "created_at": search.created_at.isoformat() if search.created_at else None
            })

    return {
        "cached_searches": count,
        "recent_searches": recent_searches
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
            "created_at": cached.created_at.isoformat() if cached.created_at else None
        }


@app.get("/cache/all")
async def get_all_cached_leads():
    """Retrieve all cached leads (for fallback when Lusha credits are exhausted)."""
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
            "message": "All cached leads retrieved"
        }


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
