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
    job_title: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    company: Optional[str] = None
    company_size: Optional[str] = None
    seniority: Optional[str] = None
    limit: int = 10

    class Config:
        extra = "allow"  # Allow additional PDL parameters


class SearchResponse(BaseModel):
    source: str  # "cache" or "api"
    count: int
    data: list


# =============================================================================
# Helper Functions
# =============================================================================

def generate_search_hash(params: dict) -> str:
    """Generate a consistent hash for search parameters."""
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_params.encode()).hexdigest()


async def fetch_from_pdl(params: dict) -> list:
    """Call People Data Labs API to search for leads using Elasticsearch query."""
    pdl_url = "https://api.peopledatalabs.com/v5/person/search"
    
    # Build Elasticsearch query from our parameters
    must_clauses = []
    
    if params.get("job_title"):
        must_clauses.append({"match": {"job_title": params["job_title"]}})
    if params.get("location"):
        # Search across multiple location fields (city, metro, region)
        must_clauses.append({
            "bool": {
                "should": [
                    {"match": {"location_locality": params["location"]}},
                    {"match": {"location_metro": params["location"]}},
                    {"match": {"location_region": params["location"]}}
                ],
                "minimum_should_match": 1
            }
        })
    if params.get("industry"):
        must_clauses.append({"term": {"industry": params["industry"]}})
    if params.get("company"):
        must_clauses.append({"match": {"job_company_name": params["company"]}})
    if params.get("company_size"):
        must_clauses.append({"term": {"job_company_size": params["company_size"]}})
    if params.get("seniority"):
        must_clauses.append({"term": {"job_title_levels": params["seniority"]}})

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

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"PDL API error: {response.text}"
        )

    result = response.json()
    return result.get("data", [])


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
    # Convert request to dict, excluding None values
    params = {k: v for k, v in request.model_dump().items() if v is not None}
    search_hash = generate_search_hash(params)

    async with async_session() as session:
        # Step 1: Check local DB
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached:
            # Step 2: Return from cache
            data = json.loads(cached.results)
            return SearchResponse(
                source="cache",
                count=len(data),
                data=data
            )

        # Step 3: Fetch from PDL API
        leads = await fetch_from_pdl(params)

        # Step 4: Cache the results
        new_cache = CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(params),
            results=json.dumps(leads)
        )
        session.add(new_cache)
        await session.commit()

        # Step 5: Return new results
        return SearchResponse(
            source="api",
            count=len(leads),
            data=leads
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
    """Get cache statistics."""
    async with async_session() as session:
        from sqlalchemy import func
        stmt = select(func.count()).select_from(CachedSearch)
        result = await session.execute(stmt)
        count = result.scalar()
    return {"cached_searches": count}


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
