"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Wiza API results to reduce costs.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field, AliasChoices
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


def f(value: str, flag: str = "i", bucket: str = None) -> dict:
    """Wiza filter value wrapper.

    Emits both the legacy `f` key (used by the prospect *list* endpoint) and the
    documented `s` key (used by the prospect *search* preview endpoint) so the
    same filter dict works against both. flag 'i' = include, 'e' = exclude.
    """
    d = {"v": value, "f": flag, "s": flag}
    if bucket:
        d["b"] = bucket
    return d


def location_filter(value: str, flag: str = "i") -> dict:
    """Build a Wiza location filter with the required 'b' (bucket) field."""
    loc = value.strip().lower()
    # Detect bucket type from the value
    if loc in _COUNTRIES:
        bucket = "country"
    elif loc in _US_STATES or (len(loc) == 2 and loc.isalpha()):
        bucket = "state"
    else:
        bucket = "city"
    return f(value, flag, bucket)


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


# =============================================================================
# Wiza Filter Builder (shared by list search, prospect preview, company search)
# =============================================================================

# Filters to drop one-by-one if Wiza rejects the request, ordered least→most useful
DROPPABLE = ["job_title_level", "job_role", "skill", "funding_stage",
             "company_size", "revenue", "company_industry", "company_location"]


# A bare hostname like "workflows.io" or "app.acme-corp.com" (no spaces, has a
# dot, ends in a TLD). Used to tell a domain apart from a plain company name.
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$", re.I)
_SUBDOMAIN_PREFIXES = {"www", "app", "mail", "go", "get", "my", "hi", "try"}


def looks_like_domain(value: str) -> bool:
    """True if `value` is a bare domain/URL rather than a company name."""
    v = value.strip()
    if not v or " " in v:
        return False
    host = v.split("//")[-1].split("/")[0].split("@")[-1]
    return bool(_DOMAIN_RE.match(host))


def company_name_from_domain(value: str) -> str:
    """Best-effort company-name token from a domain.

    'workflows.io' -> 'workflows', 'app.acme-corp.com' -> 'acme-corp'. Wiza's
    prospect filters have no company-domain field, so a domain-based "people at
    X" search is approximated with a job_company (company name) text match on
    the domain's second-level label.
    """
    host = value.strip().lower().split("//")[-1].split("/")[0].split("@")[-1]
    labels = [l for l in host.split(".") if l]
    while len(labels) > 2 and labels[0] in _SUBDOMAIN_PREFIXES:
        labels = labels[1:]
    if len(labels) >= 2:
        return labels[-2]
    return labels[0] if labels else value.strip()


def build_wiza_filters(p: dict) -> dict:
    """Translate our internal search params into a Wiza `filters` object.

    Used by both the prospect list workflow (create_prospect_list) and the
    synchronous prospect search preview (/prospects/search). Wiza's prospect
    filters have no company-domain field, so a company_domain (or a `company`
    value that is really a domain) is approximated with a job_company match on
    the domain's root label — see company_name_from_domain.
    """
    fil: dict = {}

    # Job title — free text, Wiza accepts anything here
    if p.get("job_title"):
        fil["job_title"] = [f(p["job_title"])]

    # Keywords merged into job_title search (Wiza has no generic keyword field)
    if p.get("keywords"):
        fil["job_title"] = fil.get("job_title", []) + [f(p["keywords"])]

    # Seniority — only add if we have a known Wiza level
    if p.get("seniority"):
        level = SENIORITY_MAP.get(p["seniority"].lower())
        if level:
            fil["job_title_level"] = [level]

    if p.get("departments"):
        fil["job_role"] = [f(d) for d in p["departments"]]

    if p.get("location"):
        fil["location"] = [location_filter(p["location"])]

    if p.get("company_location"):
        fil["company_location"] = [location_filter(p["company_location"])]

    # Company: prefer the name; fall back to the domain's root label since Wiza
    # can't filter prospects by domain. A `company` value that is itself a bare
    # domain (e.g. the ICP parser emitting "workflows.io") is treated as one.
    company = p.get("company")
    domain = p.get("company_domain")
    if company and not domain and looks_like_domain(company):
        domain, company = company, None
    if company:
        fil["job_company"] = [f(company)]
    elif domain:
        name = company_name_from_domain(domain)
        if name:
            fil["job_company"] = [f(name)]

    if p.get("industry"):
        fil["company_industry"] = [f(p["industry"])]

    if p.get("company_size"):
        sizes = COMPANY_SIZE_MAP.get(p["company_size"], [])
        if sizes:
            fil["company_size"] = sizes

    if p.get("technologies"):
        fil["skill"] = [f(t) for t in p["technologies"]]

    if p.get("intent_topics"):
        stages = []
        for topic in p["intent_topics"]:
            stages.extend(INTENT_TO_FUNDING.get(topic, []))
        if stages:
            fil["funding_stage"] = list(set(stages))

    if p.get("revenue_min") or p.get("revenue_max"):
        rev: dict = {}
        if p.get("revenue_min"):
            rev["min"] = p["revenue_min"]
        if p.get("revenue_max"):
            rev["max"] = p["revenue_max"]
        fil["revenue"] = rev

    return fil


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
    # Accept `text` or `query` — edge functions post `{"query": "..."}` (the same
    # key /search uses), so alias it in rather than 422-ing on the missing `text`.
    text: str = Field(validation_alias=AliasChoices("text", "query"))

    model_config = {"populate_by_name": True}


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


def transform_reveal_contact(contact: dict) -> dict:
    """Transform a Wiza *individual_reveal* response into the internal lead format.

    The reveal endpoint's schema differs from the list-contacts schema handled by
    transform_wiza_contact:

      * the person's name is in `name` (there are no first_name/last_name fields),
        whereas list contacts put the *company* name in `name`;
      * the company name is in `company`;
      * the LinkedIn URL is in `linkedin_profile_url`;
      * emails carry `email_type` rather than `type`.

    Passing a reveal object through transform_wiza_contact therefore drops
    first/last name and swaps the person's name into company_name. Reveals get
    their own transform to map every field to the right place.
    """
    full_name = contact.get("name") or contact.get("full_name")
    first_name = contact.get("first_name")
    last_name = contact.get("last_name")
    # The reveal schema has no first/last name — derive them from the full name.
    if full_name and not (first_name or last_name):
        parts = full_name.split()
        if len(parts) >= 2:
            first_name, last_name = parts[0], " ".join(parts[1:])
        elif parts:
            first_name = parts[0]

    emails = contact.get("emails") or []
    primary_email = (
        next((e.get("email") for e in emails
              if e.get("email_type") == "work" or e.get("type") == "work"), None)
        or (emails[0].get("email") if emails else None)
        or contact.get("email")
    )

    phones = contact.get("phones") or []
    primary_phone = (
        (phones[0].get("pretty_number") or phones[0].get("number")) if phones
        else contact.get("mobile_phone") or contact.get("phone_number") or contact.get("phone")
    )

    return {
        "contact_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "job_title": contact.get("title"),
        "linkedin_url": contact.get("linkedin_profile_url") or contact.get("linkedin"),
        "business_email": primary_email,
        "email_status": contact.get("email_status"),
        "phone": primary_phone,
        "location": contact.get("location"),
        "company_name": contact.get("company") or contact.get("company_name"),
        "company_domain": contact.get("domain") or contact.get("company_domain"),
        "industry": contact.get("industry"),
        "company_size": contact.get("company_size") or contact.get("size"),
        "company_revenue": contact.get("revenue"),
        "company_funding": contact.get("funding"),
        "technologies": None,
        "raw_data": contact,
    }


def generate_search_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def extract_json_object(text: str) -> dict:
    """Best-effort parse of a JSON object from an LLM response.

    Models sometimes wrap JSON in ```json fences or add a line of prose despite
    being told not to. Try a straight parse first, then fall back to slicing the
    outermost {...} span. Raises json.JSONDecodeError if no object can be found.
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


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

    limit = max(min(params.get("limit", 10), 100), 1)
    list_name = f"salesos-{generate_search_hash(params)[:8]}-{int(datetime.utcnow().timestamp())}"

    def make_body(fil: dict) -> dict:
        return {
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
            "filters": fil,
        }

    filters = build_wiza_filters(params)
    print(f"Wiza initial filters: {json.dumps(filters)}")

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ---- Step 1: Create prospect list (with progressive filter fallback) ----
        create_resp = None
        for _drop_attempt in range(len(DROPPABLE) + 1):
            body = make_body(filters)
            print(f"Wiza create_prospect_list body: {json.dumps(body)}")

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

            print(f"Wiza create response: {create_resp.text[:500]}")

            if create_resp.status_code == 429:
                raise HTTPException(status_code=429, detail="Wiza rate limit reached — please try again in a moment")

            if create_resp.status_code in (200, 201):
                break  # success

            # On error try dropping a filter and retry
            err_text = create_resp.text.lower()
            dropped = False
            for key in DROPPABLE:
                if key in filters and (key in err_text or "invalid" in err_text or "parameter" in err_text):
                    print(f"Dropping filter '{key}' and retrying")
                    del filters[key]
                    dropped = True
                    break
            if not dropped:
                # Nothing left to drop — bail with the error
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
# Wiza: Prospect Search Preview (synchronous, no enrichment/email credits)
# =============================================================================

async def wiza_prospect_search(params: dict, size: int) -> dict:
    """Synchronous prospect search preview via POST /prospects/search.

    Returns {"total": int, "profiles": [...]} where profiles carry
    full_name, job_title, job_company_name, job_company_website, industry and
    location_name. Unlike the list workflow this is instant and does not spend
    email credits — ideal for "how many match" counts and quick previews.

    Progressively drops droppable filters if Wiza rejects the request.
    """
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

    filters = build_wiza_filters(params)
    size = max(min(size, 30), 0)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = None
        for _drop_attempt in range(len(DROPPABLE) + 1):
            body = {"size": size, "filters": filters}
            print(f"Wiza prospect search body: {json.dumps(body)}")

            for attempt in range(3):
                resp = await client.post(
                    f"{WIZA_BASE}/prospects/search", headers=headers, json=body,
                )
                if resp.status_code != 429:
                    break
                wait = 2 ** attempt
                print(f"Rate limited — retrying in {wait}s")
                await asyncio.sleep(wait)

            print(f"Wiza prospect search status: {resp.status_code} {resp.text[:300]}")

            if resp.status_code == 429:
                raise HTTPException(status_code=429, detail="Wiza rate limit reached — please try again in a moment")
            if resp.status_code in (200, 201):
                return resp.json().get("data", {}) or {}

            # Error → try dropping a filter and retry
            err_text = resp.text.lower()
            dropped = False
            for key in DROPPABLE:
                if key in filters and (key in err_text or "invalid" in err_text or "parameter" in err_text):
                    print(f"Dropping filter '{key}' and retrying")
                    del filters[key]
                    dropped = True
                    break
            if not dropped:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Wiza prospect search error: {resp.text}",
                )

        # Exhausted droppable filters
        raise HTTPException(status_code=resp.status_code if resp else 502,
                            detail="Wiza prospect search failed after dropping optional filters")


# =============================================================================
# Wiza: Company Enrichment (domain / name / LinkedIn -> firmographics)
# =============================================================================

async def wiza_company_enrich(payload: dict) -> Optional[dict]:
    """Enrich a single company via POST /company_enrichments (2 credits).

    Accepts any of company_name, company_domain, company_linkedin_id,
    company_linkedin_slug. Returns the raw company object, or None if Wiza
    found no data.
    """
    headers = {
        "Authorization": f"Bearer {settings.wiza_api_key}",
        "Content-Type": "application/json",
    }

    print(f"Wiza company enrichment request: {json.dumps(payload)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = None
        for attempt in range(3):
            resp = await client.post(
                f"{WIZA_BASE}/company_enrichments", headers=headers, json=payload,
            )
            if resp.status_code != 429:
                break
            wait = 2 ** attempt
            print(f"Rate limited — retrying in {wait}s")
            await asyncio.sleep(wait)

        print(f"Wiza company enrich status: {resp.status_code} {resp.text[:300]}")

        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Wiza rate limit reached — please try again in a moment")
        if resp.status_code == 404:
            return None
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Wiza company enrichment error: {resp.text}",
            )

        data = resp.json().get("data") or {}
        # Wiza returns an empty/blank object when nothing is found
        if not data or not (data.get("company_name") or data.get("domain")):
            return None
        return data


# =============================================================================
# Company / preview transforms
# =============================================================================

def transform_preview_profile(profile: dict) -> dict:
    """Transform a /prospects/search preview profile to the internal lead shape."""
    return {
        "contact_name": profile.get("full_name"),
        "job_title": profile.get("job_title"),
        "linkedin_url": profile.get("linkedin_url"),
        "location": profile.get("location_name"),
        "company_name": profile.get("job_company_name"),
        "company_domain": profile.get("job_company_website"),
        "industry": profile.get("industry"),
        "raw_data": profile,
    }


def transform_wiza_company(c: dict) -> dict:
    """Normalize a Wiza company_enrichments object to a clean company shape."""
    return {
        "company_name": c.get("company_name"),
        "company_domain": c.get("domain"),
        "industry": c.get("company_industry"),
        "company_size": c.get("company_size_range") or c.get("company_size"),
        "employee_count": c.get("company_size"),
        "revenue_range": c.get("company_revenue_range"),
        "founded": c.get("company_founded"),
        "company_type": c.get("company_type"),
        "description": c.get("company_description"),
        "funding": c.get("company_funding"),
        "last_funding_round": c.get("company_last_funding_round"),
        "last_funding_amount": c.get("company_last_funding_amount"),
        "last_funding_at": c.get("company_last_funding_at"),
        "ticker": c.get("company_ticker"),
        "location": c.get("company_location"),
        "country": c.get("company_country"),
        "region": c.get("company_region"),
        "locality": c.get("company_locality"),
        "postal_code": c.get("company_postal_code"),
        "street": c.get("company_street"),
        "linkedin": c.get("company_linkedin"),
        "linkedin_id": c.get("company_linkedin_id"),
        "twitter": c.get("company_twitter"),
        "facebook": c.get("company_facebook"),
        "raw_data": c,
    }


def aggregate_companies(profiles: list) -> list:
    """Roll up prospect-search preview profiles into unique companies with
    a few sample contacts each. Used by /company/search."""
    companies: dict = {}
    for p in profiles:
        name = p.get("job_company_name")
        domain = p.get("job_company_website")
        key = (domain or name or "").strip().lower()
        if not key:
            continue
        if key not in companies:
            companies[key] = {
                "company_name": name,
                "company_domain": domain,
                "industry": p.get("industry"),
                "location": p.get("location_name"),
                "matched_contacts": 0,
                "sample_contacts": [],
            }
        entry = companies[key]
        entry["matched_contacts"] += 1
        if len(entry["sample_contacts"]) < 5:
            entry["sample_contacts"].append({
                "contact_name": p.get("full_name"),
                "job_title": p.get("job_title"),
                "linkedin_url": p.get("linkedin_url"),
            })
    return sorted(companies.values(), key=lambda c: c["matched_contacts"], reverse=True)


# =============================================================================
# Cache helper (generic, cache-first for any endpoint)
# =============================================================================

async def cached_or_fetch(kind: str, params: dict, fetcher) -> tuple[list, bool]:
    """Cache-first wrapper. `fetcher` is an async callable returning a LIST of
    raw dicts (always a list so /debug and /cache/* stay consistent). Returns
    (raw_results, from_cache). The cache key is namespaced by `kind`.
    """
    hashable = {"_kind": kind, **params}
    search_hash = generate_search_hash(hashable)
    print(f"[{kind}] search hash: {search_hash}")

    async with async_session() as session:
        stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
        cached = (await session.execute(stmt)).scalar_one_or_none()
        if cached:
            print(f"[{kind}] cache HIT")
            return json.loads(cached.results), True

        print(f"[{kind}] cache MISS — calling Wiza")
        raw = await fetcher()
        if not isinstance(raw, list):
            raw = [raw] if raw else []

        session.add(CachedSearch(
            search_hash=search_hash,
            search_params=json.dumps(hashable),
            results=json.dumps(raw),
        ))
        await session.commit()
        return raw, False


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

# =============================================================================
# Enrich Endpoint  (Wiza Individual Reveal)
# =============================================================================

class EnrichRequest(BaseModel):
    # Accept the common shorthands callers send alongside the canonical names so
    # a `{"profile_url": ...}` / `{"linkedin": ...}` / `{"domain": ...}` body
    # doesn't get its keys dropped and 400 as "provide ...".
    linkedin_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "linkedin_url", "profile_url", "linkedin", "linkedin_profile_url"))
    email: Optional[str] = None
    full_name: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("full_name", "name"))
    company: Optional[str] = None
    company_domain: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("company_domain", "domain"))

    model_config = {"populate_by_name": True}


@app.post("/enrich")
async def enrich_lead(request: EnrichRequest):
    """
    Enrich a single lead via Wiza Individual Reveal.
    Accepts linkedin_url, email, or full_name + company/domain.
    Polls until finished (up to 60s) then returns the enriched contact.
    """
    headers = {
        "Authorization": f"Bearer {settings.wiza_api_key}",
        "Content-Type": "application/json",
    }

    # Build the individual_reveal payload — Wiza accepts any one identifier.
    # Wiza's field for the LinkedIn URL is `profile_url` (not `linkedin_url`);
    # sending the wrong key left the reveal with no identifier and Wiza 400'd.
    if request.linkedin_url:
        reveal_data = {"profile_url": request.linkedin_url}
    elif request.email:
        reveal_data = {"email": request.email}
    elif request.full_name and (request.company or request.company_domain):
        reveal_data = {"full_name": request.full_name}
        if request.company:
            reveal_data["company"] = request.company
        if request.company_domain:
            reveal_data["domain"] = request.company_domain
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide linkedin_url, email, or full_name + company/domain"
        )

    body = {
        "individual_reveal": reveal_data,
        "enrichment_level": "partial",
    }

    print(f"Wiza individual reveal request: {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Start the reveal
        start_resp = await client.post(
            f"{WIZA_BASE}/individual_reveals",
            headers=headers,
            json=body,
        )
        print(f"Wiza reveal start status: {start_resp.status_code} {start_resp.text[:200]}")

        if start_resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Wiza rate limit — try again in a moment")
        if start_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=start_resp.status_code,
                detail=f"Wiza enrich error: {start_resp.text}",
            )

        reveal_id = start_resp.json().get("data", {}).get("id")
        if not reveal_id:
            raise HTTPException(status_code=500, detail="Wiza returned no reveal ID")

        # Poll until complete
        for attempt in range(20):
            await asyncio.sleep(3.0)
            poll_resp = await client.get(
                f"{WIZA_BASE}/individual_reveals/{reveal_id}",
                headers=headers,
            )
            if poll_resp.status_code not in (200, 201):
                break
            data = poll_resp.json().get("data", {})
            print(f"Wiza reveal poll #{attempt + 1}: status={data.get('status')}")
            if data.get("is_complete") or data.get("status") in ("finished", "failed"):
                contact = {k: v for k, v in data.items()
                           if k not in ("id", "status", "is_complete", "enrichment_level",
                                        "email_credits", "phone_credits", "export_credits", "api_credits")}
                return {
                    "success": True,
                    "enrichment_status": "complete" if data.get("status") != "failed" else "failed",
                    "lead": transform_reveal_contact(contact),
                }

        raise HTTPException(status_code=504, detail="Wiza enrichment timed out")


@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """Parse a plain-English ICP description into structured Wiza search filters."""
    if not settings.anthropic_api_key:
        return {"success": True, "filters": {"keywords": request.text}}

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

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

    try:
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            block.text for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        parsed = extract_json_object(raw)
    except anthropic.APIError as e:
        # Upstream model/API failure (auth, model not found, overload, rate limit).
        # Degrade to a keyword search instead of 500-ing so callers keep working.
        print(f"ICP parser: Anthropic API error — falling back to keyword search ({e})")
        return {"success": True, "filters": {"keywords": request.text}, "fallback": True}
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
        print(f"ICP parser: could not parse model output — falling back to keyword search ({e})")
        return {"success": True, "filters": {"keywords": request.text}, "fallback": True}

    return {"success": True, "filters": parsed}


# =============================================================================
# Search Endpoint
# =============================================================================

STRUCTURED_FIELDS = {
    "job_title", "departments", "seniority", "location", "company_location",
    "company", "company_domain", "company_size", "industry", "technologies",
    "keywords", "intent_topics", "revenue_min", "revenue_max", "job_change_days",
}


async def resolve_search_params(request: SearchRequest) -> dict:
    """Turn a SearchRequest into concrete Wiza params.

    If only a natural-language `query` was provided, parse it into structured
    filters via the ICP parser (falling back to a keyword search). Shared by
    /search, /prospects/preview and /company/search.
    """
    params = {k: v for k, v in request.model_dump().items() if v is not None and v != "" and v != []}
    print(f"=== SEARCH REQUEST ===\nFiltered params: {params}")

    raw_query = params.pop("query", None)
    if raw_query and not any(params.get(f) for f in STRUCTURED_FIELDS):
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

        if not any(params.get(f) for f in STRUCTURED_FIELDS):
            print("ICP parse yielded no filters — using raw query as keyword search")
            params["keywords"] = raw_query

        print(f"Final params: {params}")

    return params


@app.post("/search", response_model=SearchResponse)
async def search_leads(request: SearchRequest):
    """
    Cache-first lead search powered by Wiza.

    Flow:
    1. Check DB cache — return immediately if found
    2. Cache miss → Wiza 3-step: create list → poll → fetch contacts
    3. Cache and return results
    """
    params = await resolve_search_params(request)
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
# Prospect Preview Endpoint  (fast, synchronous, no email credits)
# =============================================================================

@app.post("/prospects/preview")
async def prospects_preview(request: SearchRequest):
    """
    Fast contact preview via Wiza /prospects/search.

    Returns the total number of matching prospects plus a sample of preview
    profiles — instantly and without spending email/enrichment credits. Use
    this to size an audience before committing to a full /search enrichment.
    """
    params = await resolve_search_params(request)
    size = max(min(request.limit or 10, 30), 0)

    async def fetcher():
        data = await wiza_prospect_search(params, size)
        # Store total alongside profiles so it survives caching
        return [{"_total": data.get("total", 0)}] + (data.get("profiles") or [])

    raw, from_cache = await cached_or_fetch("preview", {**params, "_size": size}, fetcher)

    total = raw[0].get("_total", 0) if raw and "_total" in raw[0] else 0
    profiles = [r for r in raw if "_total" not in r]
    leads = [transform_preview_profile(p) for p in profiles]

    return {
        "success": True,
        "source": "cache" if from_cache else "api",
        "from_cache": from_cache,
        "total": total,
        "count": len(leads),
        "leads": leads,
        "data": profiles,
    }


# =============================================================================
# Company Search Endpoint  (unique companies from prospect search)
# =============================================================================

@app.post("/company/search")
async def company_search(request: SearchRequest, enrich: bool = True, enrich_limit: int = 25):
    """
    Search for companies (not just people) via Wiza /prospects/search.

    Runs a prospect search using the firmographic/company filters, then rolls
    the matching profiles up into unique companies — each with a few sample
    contacts. Cache-first.

    By default each unique company is auto-enriched with full firmographics
    (industry, size, revenue, funding, socials) via Wiza company enrichment.
    That costs 2 API credits per company, capped at `enrich_limit` companies
    (top matches first). Pass `?enrich=false` to skip enrichment (0 credits).
    """
    params = await resolve_search_params(request)
    # Pull the max preview window so we can dedupe as many companies as possible
    size = 30
    enrich_limit = max(min(enrich_limit, 30), 0)

    async def fetcher():
        data = await wiza_prospect_search(params, size)
        companies = aggregate_companies(data.get("profiles") or [])
        if not enrich:
            return companies

        # Auto-enrich firmographics for the top companies (2 credits each).
        for company in companies[:enrich_limit]:
            ident = {}
            if company.get("company_domain"):
                ident["company_domain"] = company["company_domain"]
            elif company.get("company_name"):
                ident["company_name"] = company["company_name"]
            if not ident:
                continue

            try:
                raw = await wiza_company_enrich(ident)
            except HTTPException as exc:
                # Don't fail the whole search on one enrichment error.
                # On a rate limit, stop enriching further and return what we have.
                print(f"Company enrich failed for {ident}: {exc.detail}")
                if exc.status_code == 429:
                    break
                continue

            if raw:
                firmo = transform_wiza_company(raw)
                for k, v in firmo.items():
                    if k == "raw_data" or v is None:
                        continue
                    company[k] = v
                company["enriched"] = True

        return companies

    cache_params = {**params, "_enrich": enrich, "_enrich_limit": enrich_limit}
    companies, from_cache = await cached_or_fetch("company_search", cache_params, fetcher)

    return {
        "success": True,
        "source": "cache" if from_cache else "api",
        "from_cache": from_cache,
        "count": len(companies),
        "total": len(companies),
        "companies": companies,
        "data": companies,
    }


# =============================================================================
# Company Enrichment Endpoint  (domain / name / LinkedIn -> firmographics)
# =============================================================================

class CompanyEnrichRequest(BaseModel):
    # Accept both the canonical field names and the common shorthands callers
    # send (`domain`, `company`, `name`). Without these aliases a `{"domain": ...}`
    # or `{"company": ...}` body has its unknown keys dropped, leaving an empty
    # payload that 400s with "Provide one of ...".
    company_domain: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("company_domain", "domain"))
    company_name: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("company_name", "company", "name"))
    company_linkedin_id: Optional[str] = None
    company_linkedin_slug: Optional[str] = None

    model_config = {"populate_by_name": True}


@app.post("/company/enrich")
async def company_enrich(request: CompanyEnrichRequest):
    """
    Enrich / look up a single company by domain, name, or LinkedIn.

    This is the "domain search" — pass a company_domain (e.g. "stripe.com")
    and get back firmographics: industry, size, revenue, funding, location and
    social profiles. Cache-first; each live lookup costs 2 Wiza API credits.
    """
    payload = {k: v for k, v in request.model_dump().items() if v}
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: company_domain, company_name, company_linkedin_id, company_linkedin_slug",
        )

    async def fetcher():
        company = await wiza_company_enrich(payload)
        return [company] if company else []

    raw, from_cache = await cached_or_fetch("company_enrich", payload, fetcher)

    if not raw:
        raise HTTPException(status_code=404, detail="No company data found")

    return {
        "success": True,
        "source": "cache" if from_cache else "api",
        "from_cache": from_cache,
        "company": transform_wiza_company(raw[0]),
        "data": raw[0],
    }


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
            search_params = json.loads(cached.search_params)
            # Skip company/preview caches — this fallback returns enriched people only
            if search_params.get("_kind"):
                continue
            data = json.loads(cached.results)
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
