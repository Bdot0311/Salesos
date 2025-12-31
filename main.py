"""
Cache-First Lead Generation Proxy
A FastAPI application that caches PDL API results to reduce costs.
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
    pdl_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()


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

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class SearchRequest(BaseModel):
    job_title: Optional[str] = None  # e.g., "software engineer", "ceo", "data scientist"
    location: Optional[str] = None  # City name, e.g., "san francisco", "new york", "austin"
    industry: Optional[str] = None  # PDL canonical industry, e.g., "computer software", "financial services"
    company: Optional[str] = None  # Company name, e.g., "google", "microsoft", "amazon"
    company_size: Optional[str] = None  # "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"
    seniority: Optional[str] = None  # "cxo", "owner", "vp", "director", "partner", "senior", "manager", "entry", "training", "unpaid"
    limit: int = 10

    class Config:
        extra = "allow"  # Allow additional PDL parameters


class SearchResponse(BaseModel):
    success: bool = True
    source: str  # "cache" or "api"
    from_cache: bool = False
    count: int
    total: int
    leads: list  # Changed from 'data' to 'leads' for Edge Function compatibility
    data: list  # Keep for backwards compatibility


# =============================================================================
# Helper Functions
# =============================================================================

def transform_pdl_lead(lead: dict) -> dict:
    """Transform PDL lead data to ScoredLead format expected by frontend."""
    return {
        "job_title": lead.get("job_title", ""),
        "company_name": lead.get("job_company_name", ""),
        "company_domain": lead.get("job_company_website", "") or lead.get("job_company_linkedin_url", ""),
        "business_email": lead.get("work_email") or (lead.get("emails", [{}])[0].get("address") if lead.get("emails") else None),
        "contact_name": lead.get("full_name", ""),
        "industry": lead.get("industry", "") or lead.get("job_company_industry", ""),
        "company_size": lead.get("job_company_size", ""),
        "country": lead.get("location_country", "") or lead.get("location_name", ""),
        "linkedin_url": lead.get("linkedin_url", ""),
        # Raw PDL data for reference
        "raw_data": lead
    }


def generate_search_hash(params: dict) -> str:
    """Generate a consistent hash for search parameters."""
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_params.encode()).hexdigest()


async def fetch_from_pdl(params: dict) -> list:
    """Call People Data Labs API to search for leads using Elasticsearch query."""
    pdl_url = "https://api.peopledatalabs.com/v5/person/search"
    
    # Build Elasticsearch query from our parameters
    # PDL field reference: https://docs.peopledatalabs.com/docs/fields
    must_clauses = []
    
    if params.get("job_title"):
        # job_title: The person's current job title (String)
        must_clauses.append({"match": {"job_title": params["job_title"]}})
    
    if params.get("location"):
        # location_locality: City (String) - e.g., "san francisco"
        must_clauses.append({"match": {"location_locality": params["location"]}})
    
    if params.get("industry"):
        # industry: Canonical industry enum (String) - must match exactly
        # e.g., "computer software", "financial services", "marketing and advertising"
        must_clauses.append({"term": {"industry": params["industry"].lower()}})
    
    if params.get("company"):
        # job_company_name: Current company name (String)
        must_clauses.append({"match": {"job_company_name": params["company"]}})
    
    if params.get("company_size"):
        # job_company_size: Canonical size enum (String)
        # Valid values: "1-10", "11-50", "51-200", "201-500", "501-1000", 
        # "1001-5000", "5001-10000", "10001+"
        must_clauses.append({"term": {"job_company_size": params["company_size"]}})
    
    if params.get("seniority"):
        # job_title_levels: Array of canonical levels
        # Valid values: "cxo", "owner", "vp", "director", "partner", 
        # "senior", "manager", "entry", "training", "unpaid"
        must_clauses.append({"term": {"job_title_levels": params["seniority"].lower()}})

    if not must_clauses:
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    payload = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        "size": params.get("limit", 10)
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            pdl_url,
            headers={
                "X-Api-Key": settings.pdl_api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30.0
        )

    result = response.json()
    
    # Handle PDL responses
    if response.status_code == 200:
        return result.get("data", [])
    elif response.status_code == 404:
        # No results found - return empty list instead of error
        return []
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"PDL API error: {response.text}"
        )


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Cache-First Lead Generation Proxy",
    description="Proxy that caches PDL API results to reduce costs",
    version="1.0.0"
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
    await init_db()


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
    3. If not found, fetch from PDL API
    4. Cache the results
    5. Return the results
    """
    # Convert request to dict, excluding None AND empty string values
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != ""}
    search_hash = generate_search_hash(params)

    async with async_session() as session:
        # Step 1: Check local DB
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached:
            # Step 2: Return from cache
            data = json.loads(cached.results)
            leads = [transform_pdl_lead(lead) if "job_title" in lead and "full_name" not in lead.get("contact_name", "") else lead for lead in data]
            return SearchResponse(
                success=True,
                source="cache",
                from_cache=True,
                count=len(leads),
                total=len(leads),
                leads=leads,
                data=data
            )

        # Step 3: Fetch from PDL API
        raw_leads = await fetch_from_pdl(params)
        
        # Transform leads to expected format
        leads = [transform_pdl_lead(lead) for lead in raw_leads]

        # Step 4: Cache the results (store raw PDL data)
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


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics and recent searches."""
    async with async_session() as session:
        from sqlalchemy import func, desc
        
        # Get count
        count_stmt = select(func.count()).select_from(CachedSearch)
        count_result = await session.execute(count_stmt)
        count = count_result.scalar()
        
        # Get recent searches (last 10)
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
        leads = [transform_pdl_lead(lead) for lead in data]
        
        return {
            "success": True,
            "from_cache": True,
            "search_params": json.loads(cached.search_params),
            "leads": leads,
            "data": data,
            "count": len(leads),
            "total": len(leads),
            "created_at": cached.created_at.isoformat() if cached.created_at else None
        }


@app.get("/cache/all")
async def get_all_cached_leads():
    """Retrieve all cached leads (for fallback when PDL credits exhausted)."""
    async with async_session() as session:
        stmt = select(CachedSearch).order_by(CachedSearch.created_at.desc())
        result = await session.execute(stmt)
        all_cached = result.scalars().all()
        
        all_leads = []
        seen_contacts = set()
        
        for cached in all_cached:
            data = json.loads(cached.results)
            for lead in data:
                # Deduplicate by full_name + company
                key = f"{lead.get('full_name', '')}-{lead.get('job_company_name', '')}"
                if key not in seen_contacts:
                    seen_contacts.add(key)
                    all_leads.append(transform_pdl_lead(lead))
        
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
