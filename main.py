"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Wiza API results to reduce costs.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field, AliasChoices
from pydantic_settings import BaseSettings
from sqlalchemy import Column, String, Text, DateTime, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    database_url: str
    wiza_api_key: str
    anthropic_api_key: Optional[str] = None
    # Prospecting data providers, tried in order: Bytemine, then Crustdata, then
    # Wiza. A provider is only in the chain if its key is configured, so adding a
    # key is all it takes to put one in front. SEARCH_PROVIDER pins whichever
    # provider should be tried first; the others still follow it as fallbacks.
    #
    # The order is by capability, not preference. Bytemine and Crustdata both
    # return masked/flagged contacts from search and charge on reveal, which
    # matches how the app bills. Wiza's list workflow returns enriched contacts
    # and spends an email credit per search, so it sits last.
    bytemine_api_key: Optional[str] = None
    crustdata_api_key: Optional[str] = None
    search_provider: str = "bytemine"
    # ICP parsing is rule-based by default (no API credits needed). Set
    # USE_LLM_PARSER=true (with ANTHROPIC_API_KEY) to also run the LLM and let it
    # fill any gaps the rules miss.
    use_llm_parser: bool = False
    # Search results older than this are treated as cache misses. Set to 0 to
    # disable search-result caching without dropping the cache table.
    search_cache_ttl_seconds: int = 3600

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
CRUSTDATA_BASE = "https://api.crustdata.com"
CRUSTDATA_VERSION = "2025-11-01"
# Bytemine routes every endpoint through one gateway: the real path and method
# travel in the JSON body rather than the URL.
BYTEMINE_GATEWAY = "https://bvjmtgaxijpyasjtaqiv.supabase.co/functions/v1/api-gateway"

# Default order, best-fit first. Filtered down to configured providers by
# provider_chain().
PROVIDER_ORDER = ("bytemine", "crustdata", "wiza")


def provider_configured(name: str) -> bool:
    """True when this provider has the credentials to be called at all."""
    if name == "bytemine":
        return bool(settings.bytemine_api_key)
    if name == "crustdata":
        return bool(settings.crustdata_api_key)
    if name == "wiza":
        return bool(settings.wiza_api_key)
    return False


def provider_chain() -> list[str]:
    """Providers to try for one search, in order.

    SEARCH_PROVIDER pins which one leads; the rest follow as fallbacks in
    PROVIDER_ORDER. Only configured providers appear, so an unset key removes a
    provider from the chain rather than breaking the request — and a chain of
    one behaves exactly like the single-provider setup this replaced.
    """
    preferred = (settings.search_provider or PROVIDER_ORDER[0]).strip().lower()
    ordered = [preferred] + [p for p in PROVIDER_ORDER if p != preferred]
    chain = [p for p in ordered if p in PROVIDER_ORDER and provider_configured(p)]
    # Wiza's key is required by Settings, so the chain is never empty in
    # practice; the guard keeps a misconfigured environment from failing here
    # with an IndexError instead of a readable provider error.
    return chain or ["wiza"]


def provider_state() -> tuple[str, bool]:
    """(provider, degraded) — the provider that leads the chain.

    `degraded` means the provider SEARCH_PROVIDER asked for is not the one that
    will run, because its key is missing. That distinguishes an accident from
    someone deliberately setting SEARCH_PROVIDER=wiza, and the two must not
    behave alike. Wiza's list
    workflow returns already-enriched contacts and spends an email credit per
    search, so a silent degradation would bill every search to the Wiza account
    while the app charges the user nothing — searching is free now, the credit
    is spent on reveal. The degraded path therefore searches through Wiza's
    preview endpoint, which spends no credits and returns identity + company +
    LinkedIn URL, leaving the reveal to /enrich exactly as Crustdata does.
    """
    chain = provider_chain()
    preferred = (settings.search_provider or PROVIDER_ORDER[0]).strip().lower()
    head = chain[0]
    # Degraded means the provider that was asked for is not the one that will
    # run, which for Wiza changes what a search costs — see the note above.
    return head, head != preferred


def active_provider() -> str:
    """Which data provider to use for this request."""
    return provider_state()[0]


def provider_degraded() -> bool:
    """True when we fell back to Wiza because Crustdata has no key configured."""
    return provider_state()[1]


# =============================================================================
# Wiza Filter Mappings
# =============================================================================

# Our seniority keys -> Wiza job_title_level enum.
# Valid values (per Wiza prospect-search docs): CXO, VP, Director, Manager,
# Senior, Entry, Owner, Partner, Training, Unpaid.
SENIORITY_MAP = {
    "entry":    "Entry",
    "training": "Training",
    "intern":   "Entry",
    "junior":   "Entry",     # Wiza has no "Junior" level; Entry is closest
    "senior":   "Senior",
    "manager":  "Manager",
    "head":     "Manager",
    "director": "Director",
    "partner":  "Partner",
    "vp":       "VP",
    "c_suite":  "CXO",
    "cxo":      "CXO",
    "owner":    "Owner",
    "founder":  "Owner",     # Wiza has no "Founder" level; Owner is closest
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

# Buyer intent -> Wiza funding_stage.v values (valid: pre_seed, seed, series_a,
# series_b, series_c, series_d, series_e-j, other). IPO and Mergers aren't
# funding stages — they map to company_type / funding_type instead (see
# build_wiza_filters).
INTENT_TO_FUNDING = {
    "Funding":    ["pre_seed", "seed", "series_a", "series_b", "series_c",
                   "series_d", "series_e-j"],
    "Investment": ["seed", "series_a", "series_b"],
}

# Company revenue buckets (lo inclusive, hi exclusive) -> Wiza revenue enum.
REVENUE_BUCKETS = [
    (0,        1_000_000,   "$0-$1M"),
    (1_000_000, 10_000_000, "$1M-$10M"),
    (10_000_000, 25_000_000, "$10M-$25M"),
    (25_000_000, 50_000_000, "$25M-$50M"),
    (50_000_000, 100_000_000, "$50M-$100M"),
    (100_000_000, 250_000_000, "$100M-$250M"),
    (250_000_000, 500_000_000, "$250M-$500M"),
    (500_000_000, 1_000_000_000, "$500M-$1B"),
    (1_000_000_000, 10_000_000_000, "$1B-$10B"),
    (10_000_000_000, float("inf"), "$10B+"),
]


def revenue_buckets(rmin, rmax) -> list:
    """Map a min/max USD revenue range to the overlapping Wiza revenue enums."""
    lo_bound = rmin if rmin is not None else 0
    hi_bound = rmax if rmax is not None else float("inf")
    return [label for lo, hi, label in REVENUE_BUCKETS
            if hi > lo_bound and lo < hi_bound]


# Our department labels -> Wiza job_role enum (plain lowercase strings).
JOB_ROLE_MAP = {
    "sales": "sales", "marketing": "marketing", "engineering": "engineering",
    "finance": "finance", "legal": "legal", "operations": "operations",
    "design": "design", "media": "media", "education": "education",
    "health": "health", "healthcare": "health", "trades": "trades",
    "human resources": "human_resources", "hr": "human_resources",
    "real estate": "real_estate", "public relations": "public_relations",
    "pr": "public_relations", "customer service": "customer_service",
    "customer support": "customer_service", "support": "customer_service",
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
    """Build a Wiza location filter with the required 'b' (bucket) field.

    Wiza wants the value shaped by bucket: country = 'country', state =
    'state, country', city = 'city, state, country'. We can reliably qualify a
    US state ('california' -> 'california, united states'); country is passed
    through and a city is best-effort (Wiza still matches many bare cities).
    """
    value = value.strip()
    loc = value.lower()
    if loc in _COUNTRIES:
        bucket = "country"
    elif loc in _US_STATES or (len(loc) == 2 and loc.isalpha()):
        bucket = "state"
        if loc in _US_STATES and "," not in value:
            value = f"{value}, United States"
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
# Filters Wiza may reject outright and that can be dropped without changing who
# the search is for. Retrying without one of these recovers a request Wiza would
# otherwise refuse.
#
# company_size, company_industry and company_location are deliberately NOT here.
# They define the ICP, and dropping them silently returns leads from the wrong
# size, the wrong industry or the wrong country while reporting success — the
# same failure the frontend's progressive fallback used to produce. If Wiza
# refuses one of those, the search fails loudly and the caller sees why.
DROPPABLE = ["job_title_level", "job_role", "skill", "funding_stage", "revenue"]

# Kept for the error message so a refusal names the filter Wiza actually
# rejected rather than failing anonymously.
ICP_DEFINING = ("company_size", "company_industry", "company_location")


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


def find_domain_in_text(text: str) -> Optional[str]:
    """Return the first bare domain found in free text, else None.

    Lets a query like "people who work at workflows.io" route to a company
    filter even when the ICP parser is unavailable (e.g. no LLM credits),
    instead of degrading to a keyword/job-title search that never matches.
    """
    for token in re.split(r"[\s,;]+", text.strip()):
        t = token.strip(".,;:!?()[]\"'")
        if looks_like_domain(t):
            return t
    return None


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


def domain_host(value: str) -> str:
    """The registrable host of a domain, TLD kept and subdomains stripped.

    'https://www.workflows.io/x' -> 'workflows.io', 'app.acme-corp.com' ->
    'acme-corp.com'. Used as a job_company candidate because many companies are
    named after their full domain (e.g. the brand is literally 'workflows.io').
    """
    host = value.strip().lower().split("//")[-1].split("/")[0].split("@")[-1]
    labels = [l for l in host.split(".") if l]
    while len(labels) > 2 and labels[0] in _SUBDOMAIN_PREFIXES:
        labels = labels[1:]
    return ".".join(labels) if labels else value.strip().lower()


# =============================================================================
# Rule-based ICP parser (deterministic, no LLM / API credits required)
# =============================================================================

# Country/UK/US abbreviations -> canonical location string.
_LOCATION_ALIASES = {
    "united states": "United States", "u.s.a": "United States", "u.s": "United States",
    "usa": "United States", "us": "United States", "america": "United States",
    "united kingdom": "United Kingdom", "u.k": "United Kingdom", "uk": "United Kingdom",
    "britain": "United Kingdom", "england": "United Kingdom",
}

# Industry keyword -> a Wiza company_industry value. Values MUST come from Wiza's
# documented industry vocabulary (all lowercase). Only high-confidence mappings;
# unknown words are left for the keyword fallback.
_INDUSTRY_MAP = {
    "fintech": "financial services", "financial services": "financial services",
    "finance": "financial services", "banking": "banking",
    "saas": "computer software", "software": "computer software",
    "ai": "computer software", "artificial intelligence": "computer software",
    "machine learning": "computer software",
    "healthcare": "hospital & health care", "health care": "hospital & health care",
    "healthtech": "hospital & health care", "biotech": "biotechnology",
    "biotechnology": "biotechnology", "pharma": "pharmaceuticals",
    "edtech": "e-learning", "education": "education management",
    "ecommerce": "internet", "e-commerce": "internet",
    "cybersecurity": "computer & network security",
    "cyber security": "computer & network security",
    "security": "computer & network security",
    "insurance": "insurance", "insurtech": "insurance",
    "real estate": "real estate", "proptech": "real estate",
    "retail": "retail",
    "marketing": "marketing and advertising", "advertising": "marketing and advertising",
    "media": "media production", "gaming": "computer games", "games": "computer games",
    "crypto": "financial services", "web3": "financial services", "blockchain": "financial services",
    "telecom": "telecommunications", "telecommunications": "telecommunications",
    "logistics": "logistics and supply chain", "supply chain": "logistics and supply chain",
    "hospitality": "hospitality",
    "automotive": "automotive", "energy": "oil & energy", "consulting": "management consulting",
    "nonprofit": "non-profit organization management",
}

# Query terms no provider taxonomy can express.
#
# All three providers classify companies with LinkedIn's industry list, which
# has no AI category — the nearest value is "computer software", meaning every
# software company there is. Mapping "AI SaaS" onto it and stopping there is how
# a search for AI companies returned CEOs at Boeing and VC firms: the word that
# defined the ICP was spent on a filter that cannot carry it.
#
# A term listed here keeps its taxonomy value as a supporting filter *and* is
# emitted as `keywords`. Every provider must then either search it as free text
# or refuse the search outright, so the chain moves to one that can — see
# ProviderUnsupported.
_SEMANTIC_ONLY_TERMS = (
    "artificial intelligence", "computer vision", "machine learning",
    "deep learning", "generative ai", "ai saas", "gen ai", "genai", "mlops",
    "ai/ml", "llms", "llm", "nlp", "ai", "ml",
)


def semantic_terms_in(text: str) -> list[str]:
    """The parts of a query that no industry taxonomy can express.

    Matched longest-first so "ai saas" wins over the bare "ai" inside it, and
    with non-alphanumeric boundaries rather than \\b so that "email", "chair" and
    "retail" do not read as an AI query — which is exactly the collapse that
    made every "AI SaaS" search return generic software leads.
    """
    low = (text or "").lower()
    found: list[str] = []
    for term in sorted(_SEMANTIC_ONLY_TERMS, key=len, reverse=True):
        if not re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            continue
        # A longer match already covers this one ("ai" inside "ai saas").
        if any(term in seen for seen in found):
            continue
        found.append(term)
    return found


class ProviderUnsupported(Exception):
    """This provider cannot express one of the requested filters.

    Raised instead of quietly dropping it: the chain moves to the next provider,
    which is the whole reason a fallback exists. Silently ignoring the filter
    would return leads outside the requested ICP.
    """

    def __init__(self, field: str, value):
        super().__init__(f"{field}={value!r}")
        self.field = field
        self.value = value


# Wiza's complete, fixed company_industry vocabulary (from the prospect-search
# docs). Anything outside this set is silently ignored by Wiza, so we validate
# against it before sending — see normalize_industry.
WIZA_INDUSTRIES = frozenset({
    "accounting", "airlines/aviation", "alternative dispute resolution",
    "alternative medicine", "animation", "apparel & fashion",
    "architecture & planning", "arts and crafts", "automotive",
    "aviation & aerospace", "banking", "biotechnology", "broadcast media",
    "building materials", "business supplies and equipment", "capital markets",
    "chemicals", "civic & social organization", "civil engineering",
    "commercial real estate", "computer & network security", "computer games",
    "computer hardware", "computer networking", "computer software",
    "construction", "consumer electronics", "consumer goods",
    "consumer services", "cosmetics", "dairy", "defense & space", "design",
    "e-learning", "education management", "electrical/electronic manufacturing",
    "entertainment", "environmental services", "events services",
    "executive office", "facilities services", "farming", "financial services",
    "fine art", "fishery", "food & beverages", "food production", "fund-raising",
    "furniture", "gambling & casinos", "government administration",
    "government relations", "graphic design", "health, wellness and fitness",
    "higher education", "hospital & health care", "hospitality",
    "human resources", "import and export", "individual & family services",
    "industrial automation", "information services",
    "information technology and services", "insurance", "international affairs",
    "international trade and development", "internet", "investment banking",
    "investment management", "judiciary", "law enforcement", "law practice",
    "legal services", "legislative office", "libraries",
    "logistics and supply chain", "luxury goods & jewelry", "machinery",
    "management consulting", "maritime", "market research",
    "marketing and advertising", "mechanical or industrial engineering",
    "media production", "medical devices", "medical practice",
    "mental health care", "military", "mining & metals",
    "motion pictures and film", "museums and institutions", "music",
    "nanotechnology", "newspapers", "non-profit organization management",
    "oil & energy", "online media", "outsourcing/offshoring",
    "package/freight delivery", "packaging and containers",
    "paper & forest products", "performing arts", "pharmaceuticals",
    "philanthropy", "photography", "plastics", "political organization",
    "primary/secondary education", "printing",
    "professional training & coaching", "program development", "public policy",
    "public relations and communications", "public safety", "publishing",
    "railroad manufacture", "ranching", "real estate",
    "recreational facilities and services", "religious institutions",
    "renewables & environment", "research", "restaurants", "retail",
    "security and investigations", "semiconductors", "shipbuilding",
    "sporting goods", "sports", "staffing and recruiting", "supermarkets",
    "telecommunications", "textiles", "think tanks", "tobacco",
    "translation and localization", "transportation/trucking/railroad",
    "utilities", "venture capital & private equity", "veterinary",
    "warehousing", "wholesale", "wine and spirits", "wireless",
    "writing and editing",
})

# Common free-form phrasings (often from the LLM parser) that aren't in
# _INDUSTRY_MAP and don't exactly match Wiza's wording -> the Wiza value.
_INDUSTRY_ALIASES = {
    "software development": "computer software",
    "information technology": "information technology and services",
    "it": "information technology and services",
    "it services": "information technology and services",
    "tech": "computer software", "technology": "computer software",
    "health and wellness": "health, wellness and fitness",
    "health & wellness": "health, wellness and fitness",
    "wellness": "health, wellness and fitness",
    "fitness": "health, wellness and fitness",
    "venture capital": "venture capital & private equity",
    "private equity": "venture capital & private equity",
    "vc": "venture capital & private equity",
    "recruiting": "staffing and recruiting",
    "staffing": "staffing and recruiting",
    "manufacturing": "mechanical or industrial engineering",
    "transportation": "transportation/trucking/railroad",
    "food and beverage": "food & beverages",
    "food & beverage": "food & beverages",
    "aerospace": "aviation & aerospace",
    "defense": "defense & space",
    "renewables": "renewables & environment",
    "renewable energy": "renewables & environment",
    "clean energy": "renewables & environment",
}


def normalize_industry(value: str):
    """Coerce an arbitrary industry string to a valid Wiza company_industry value.

    Wiza silently ignores industries outside its fixed vocabulary, and the
    `industry` field can arrive free-form (Title-Case, out-of-vocab) from the
    LLM parser as well as from the rule-based _INDUSTRY_MAP. Snap it onto the
    documented set. Returns the canonical lowercase value, or None when we can't
    map it confidently — the caller then drops the filter rather than send an
    invalid value that Wiza would ignore.
    """
    if not value:
        return None
    key = value.strip().lower()
    if not key:
        return None
    if key in WIZA_INDUSTRIES:      # already a valid Wiza industry
        return key
    if key in _INDUSTRY_MAP:        # keyword/synonym (fintech -> financial services)
        return _INDUSTRY_MAP[key]
    return _INDUSTRY_ALIASES.get(key)  # common phrasing, else None


# Named company-size tiers -> Wiza size bucket.
_SIZE_TIERS = {
    "smb": "11-50", "small business": "11-50", "small businesses": "11-50",
    "mid-market": "201-500", "midmarket": "201-500", "mid market": "201-500",
    "enterprise": "1001-5000", "enterprises": "1001-5000",
    "large enterprise": "5001-10000",
}

# Detection regex -> our seniority key (ordered strongest first; first hit wins).
_SENIORITY_RULES = [
    (r"\bco-?founders?\b|\bfounders?\b", "founder"),
    (r"\bowners?\b", "owner"),
    (r"\b(c[e-z]?os?|ceos?|ctos?|cfos?|coos?|cmos?|cios?|cisos?|cros?|cpos?|chros?|ccos?|cdos?|csos?|c-suite|chief)\b", "c_suite"),
    (r"\b(svps?|evps?|vps?|vice presidents?)\b", "vp"),
    (r"\bdirectors?\b", "director"),
    (r"\b(heads?|managers?)\b", "manager"),
    (r"\b(senior|sr\.?)\b", "senior"),
    (r"\b(junior|jr\.?|entry[- ]level|interns?)\b", "junior"),
]

_CSUITE_ACRONYMS = r"\b(ceo|cto|cfo|coo|cmo|cio|ciso|cro|cpo|chro|cco|cdo|cso)s?\b"

# Two-word "<function> <role>" and single role nouns for job_title detection.
_ROLE_NOUNS = ("engineer", "developer", "designer", "manager", "director",
               "analyst", "lead", "specialist", "representative", "executive",
               "recruiter", "consultant", "architect", "scientist", "accountant",
               "controller", "marketer", "administrator", "officer", "president")
_TITLE_PHRASES = sorted(
    ["software engineer", "account executive", "account manager", "product manager",
     "project manager", "program manager", "data scientist", "data analyst",
     "business analyst", "financial analyst", "sales development representative",
     "business development representative", "solutions engineer", "sales engineer",
     "product designer", "full stack developer", "co-founder", "cofounder",
     "founder", "owner", "president", "recruiter", "controller", "accountant",
     "consultant", "architect", "engineer", "developer", "designer", "sdr", "bdr"],
    key=len, reverse=True,
)

_DEPARTMENTS = {"sales": "Sales", "marketing": "Marketing", "engineering": "Engineering",
                "product": "Product", "finance": "Finance", "hr": "Human Resources",
                "human resources": "Human Resources", "legal": "Legal",
                "operations": "Operations", "design": "Design", "it": "Information Technology",
                "data": "Data", "support": "Customer Support"}


def _size_bucket(n: int) -> str:
    """Map an employee count to the nearest Wiza size bucket."""
    for ceiling, key in [(10, "1-10"), (50, "11-50"), (200, "51-200"), (500, "201-500"),
                         (1000, "501-1000"), (5000, "1001-5000"), (10000, "5001-10000")]:
        if n <= ceiling:
            return key
    return "10001+"


def _money(num: str, unit: str) -> int:
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6,
            "b": 1e9, "billion": 1e9}.get(unit.lower(), 1)
    return int(float(num.replace(",", "")) * mult)


def heuristic_parse_icp(text: str) -> dict:
    """Parse a plain-English ICP into Wiza search filters — deterministically,
    with no LLM call. Covers the common phrasings (title, seniority, location,
    company/domain, size, industry, revenue, buyer intent, job change). Anything
    it can't structure falls through to a keyword search.
    """
    orig = (text or "").strip()
    low = orig.lower()
    filters: dict = {}

    # --- Company domain / name -------------------------------------------------
    domain = find_domain_in_text(orig)
    if domain:
        filters["company_domain"] = domain
    else:
        m = re.search(r"(?:\bwork(?:s|ing)?\s+)?(?:\bat\b|@)\s+"
                      r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})", orig)
        if m:
            cand = re.sub(r"\b(compan(?:y|ies)|startups?|inc\.?|llc|corp\.?)\b.*$", "",
                          m.group(1), flags=re.I).strip()
            cl = cand.lower()
            # "VP Sales at AI SaaS startups" is not a request for a company
            # named "AI SaaS" — the phrase after "at" can just as easily be the
            # segment, and searching it as a company name matches near-nothing.
            if cand and cl not in _INDUSTRY_MAP and cl not in _COUNTRIES \
                    and cl not in _US_STATES and cl not in _LOCATION_ALIASES \
                    and not semantic_terms_in(cand):
                filters["company"] = cand

    # --- Location (people location; "hq/headquartered" -> company_location) ----
    loc = None
    for alias, canon in _LOCATION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low):
            loc = canon
            break
    if not loc:
        for name in sorted(_COUNTRIES | _US_STATES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", low):
                loc = name.title()
                break
    if not loc:
        m = re.search(r"\b(?:in|near|based in|located in|from)\s+"
                      r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", orig)
        if m and m.group(1).lower() not in _INDUSTRY_MAP:
            loc = m.group(1).strip()
    if loc:
        if re.search(r"\b(hq|headquarter(?:ed|s)?|based out of)\b", low):
            filters["company_location"] = loc
        else:
            filters["location"] = loc

    # --- Seniority -------------------------------------------------------------
    for pattern, key in _SENIORITY_RULES:
        if re.search(pattern, low):
            filters["seniority"] = key
            break

    # --- Job title -------------------------------------------------------------
    title = None
    m = re.search(r"\b(svp|evp|vp|vice president|head|director|manager|lead|chief)\s+of\s+"
                  r"[a-z][a-z ]*?(?=\s+(?:at|in|for|with|who|that|based|located)\b|[.,]|$)", low)
    if m:
        title = orig[m.start():m.end()].strip()
    if not title:
        m = re.search(_CSUITE_ACRONYMS, low)
        if m:
            title = m.group(1).upper()  # group drops any trailing plural 's'
    if not title:
        m = re.search(r"\b([a-z]+)\s+(" + "|".join(_ROLE_NOUNS) + r")s?\b", low)
        if m:
            title = orig[m.start():m.end()].strip()
    if not title:
        for phrase in _TITLE_PHRASES:
            mm = re.search(rf"\b{re.escape(phrase)}s?\b", low)
            if mm:
                title = orig[mm.start():mm.end()].strip()
                break
    if title:
        filters["job_title"] = title
        # Derive a department from a "<X> of <Dept>" title.
        dm = re.search(r"\bof\s+([a-z]+)", title.lower())
        if dm and dm.group(1) in _DEPARTMENTS:
            filters["departments"] = [_DEPARTMENTS[dm.group(1)]]

    # --- Industry (skip when the word is a role modifier, e.g. "software
    #     engineer", "marketing director" — that's a title/department, not an
    #     industry filter) --------------------------------------------------------
    _role_alt = "|".join(_ROLE_NOUNS)
    for kw in sorted(_INDUSTRY_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(kw)}\b(?!\s+(?:{_role_alt})s?\b)", low):
            filters["industry"] = _INDUSTRY_MAP[kw]
            break

    # --- Terms the taxonomy cannot carry --------------------------------------
    # "AI SaaS" resolves to industry=computer software above, which is every
    # software company on earth. Keep that as a supporting filter but also send
    # the defining words as free text, so a provider either searches them or
    # refuses — see _SEMANTIC_ONLY_TERMS.
    semantic = semantic_terms_in(orig)
    if semantic:
        filters["keywords"] = " ".join(semantic)

    # --- Company size ----------------------------------------------------------
    size = None
    for tier in sorted(_SIZE_TIERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(tier)}\b", low):
            size = _SIZE_TIERS[tier]
            break
    if not size:
        m = re.search(r"(\d[\d,]*)\s*(?:-|to|–)\s*(\d[\d,]*)\s*"
                      r"(?:employees|people|person|headcount|staff|team)", low)
        if m:
            size = _size_bucket(int(m.group(2).replace(",", "")))
        else:
            m = re.search(r"(\d[\d,]*)\s*\+?\s*"
                          r"(?:employees|people|person|headcount|staff)", low)
            if m:
                size = _size_bucket(int(m.group(1).replace(",", "")))
    if size:
        filters["company_size"] = size

    # --- Revenue (only when explicitly about revenue, not funding amounts) -----
    if re.search(r"\b(revenue|arr|turnover)\b", low):
        money = r"\$?\s*(\d[\d,.]*)\s*(k|m|b|thousand|million|billion)\b"
        between = re.search(r"between\s+" + money + r"\s+and\s+" + money, low)
        if between:
            filters["revenue_min"] = _money(between.group(1), between.group(2))
            filters["revenue_max"] = _money(between.group(3), between.group(4))
        else:
            m = re.search(r"(over|above|more than|greater than|>|at least)\s+" + money, low)
            if m:
                filters["revenue_min"] = _money(m.group(2), m.group(3))
            m = re.search(r"(under|below|less than|<|up to)\s+" + money, low)
            if m:
                filters["revenue_max"] = _money(m.group(2), m.group(3))
            if "revenue_min" not in filters and "revenue_max" not in filters:
                m = re.search(money, low)
                if m:
                    filters["revenue_min"] = _money(m.group(1), m.group(2))

    # --- Buyer intent ----------------------------------------------------------
    intent = []
    if re.search(r"\b(series [a-f]|seed|pre-?seed|raised|funding|funded|"
                 r"venture[- ]backed|vc[- ]backed)\b", low):
        intent.append("Funding")
    if re.search(r"\b(ipo|going public|public offering)\b", low):
        intent.append("IPO")
    if re.search(r"\b(merger|m&a|acquisitions?|acquired)\b", low):
        intent.append("Mergers")
    if intent:
        filters["intent_topics"] = intent

    # --- Recent job change -----------------------------------------------------
    if re.search(r"\b(recently (changed|joined|started)|new (role|job|position)|"
                 r"just (joined|started)|changed jobs|job change)\b", low):
        filters["job_change_days"] = 90

    # --- Fallback --------------------------------------------------------------
    if not filters:
        filters["keywords"] = orig

    return filters


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

    # Wiza has no free-text field. Merging keywords into job_title looked like a
    # workaround but did the opposite of narrowing: Wiza ORs the values in a
    # filter, so job_title=["Founder", "AI SaaS"] matches every Founder
    # anywhere — which is how an AI search returned founders at aerospace and
    # venture firms. Refuse, and let the chain reach a provider with free text.
    if p.get("keywords"):
        raise ProviderUnsupported("keywords", p["keywords"])

    # Seniority — only add if we have a known Wiza level
    if p.get("seniority"):
        level = SENIORITY_MAP.get(p["seniority"].lower())
        if level:
            fil["job_title_level"] = [level]

    # job_role is a plain enum array (not {v,s} objects); map our labels to
    # Wiza's role vocabulary and drop anything that isn't a valid role.
    if p.get("departments"):
        roles = [JOB_ROLE_MAP[d.strip().lower()] for d in p["departments"]
                 if d.strip().lower() in JOB_ROLE_MAP]
        if roles:
            fil["job_role"] = list(dict.fromkeys(roles))  # dedupe, keep order

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
    # Build OR candidates for the company-name filter. For a domain we can't be
    # sure how the company is recorded, so offer several forms — the resolved
    # name (if any), the full domain ("workflows.io", for domain-named brands),
    # and the root label ("workflows"). Wiza treats multiple job_company values
    # as OR, so any one matching surfaces the company in a single search.
    candidates = []
    if company:
        candidates.append(company)
    if domain:
        candidates.append(domain_host(domain))
        candidates.append(company_name_from_domain(domain))
    seen, values = set(), []
    for c in candidates:
        key = (c or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            values.append(c)
    if values:
        fil["job_company"] = [f(c) for c in values]

    # company_industry must be one of Wiza's fixed values or Wiza ignores it.
    # An unmappable industry used to be dropped with a log line, which turned a
    # narrow request into a broad search without telling anyone. Refuse instead.
    if p.get("industry"):
        industry = normalize_industry(p["industry"])
        if not industry:
            raise ProviderUnsupported("industry", p["industry"])
        fil["company_industry"] = [f(industry)]

    if p.get("company_size"):
        sizes = COMPANY_SIZE_MAP.get(p["company_size"], [])
        if sizes:
            fil["company_size"] = sizes

    # skill is a plain string array, not {v,s} objects.
    if p.get("technologies"):
        fil["skill"] = list(p["technologies"])

    # Buyer intent. Funding/Investment -> funding_stage ({t, v:[stages]}); IPO ->
    # company_type "public"; Mergers -> funding_type private_equity.
    if p.get("intent_topics"):
        topics = p["intent_topics"]
        stages = []
        for topic in topics:
            stages.extend(INTENT_TO_FUNDING.get(topic, []))
        if stages:
            fil["funding_stage"] = {"t": "any", "v": list(dict.fromkeys(stages))}
        if "IPO" in topics:
            fil["company_type"] = list(dict.fromkeys(fil.get("company_type", []) + ["public"]))
        if "Mergers" in topics:
            fil["funding_type"] = {"t": "any", "v": ["private_equity"]}

    # Revenue -> overlapping Wiza revenue enum buckets (not a min/max object).
    if p.get("revenue_min") or p.get("revenue_max"):
        buckets = revenue_buckets(p.get("revenue_min"), p.get("revenue_max"))
        if buckets:
            fil["revenue"] = buckets

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


class CampaignSeenLead(Base):
    """A durable per-campaign ledger used to prevent resurfacing people."""

    __tablename__ = "campaign_seen_leads"

    campaign_id = Column(String(128), primary_key=True)
    lead_key = Column(String(256), primary_key=True)
    crustdata_person_id = Column(String(64), nullable=True)
    profile_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


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

    # Pagination. Callers page either by number or by cursor; both are accepted
    # because the two halves of the stack disagreed about which to use.
    #
    # The caller sent `offset` and this model did not declare it, so pydantic
    # dropped it: page 2 arrived byte-identical to page 1, produced the same
    # cache key, and was answered from page 1's cached rows. Every page showed
    # the same leads, on every search, and no live call was ever made.
    page: int = 1
    offset: Optional[int] = None

    # Search controls. Cursor is opaque and must be echoed from `next_cursor`.
    cursor: Optional[str] = None
    campaign_id: Optional[str] = None
    exclude_profiles: Optional[list[str]] = None
    refresh: bool = False
    # Broad title-only searches are blocked by default because they were the
    # final, unsafe step in the frontend's progressive fallback. A deliberate
    # title-only search must opt in explicitly.
    allow_broad_search: bool = False

    @property
    def start_offset(self) -> int:
        """Rows to skip, however the caller expressed it."""
        if self.offset is not None:
            return max(self.offset, 0)
        return max(self.page - 1, 0) * max(self.limit, 1)

    @field_validator("keywords", "seniority", "job_title", "location", "company_location",
                     "company", "company_domain", "industry", "company_size", "query", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return v

    @field_validator("technologies", "departments", "intent_topics", "signals",
                     "exclude_profiles", mode="before")
    @classmethod
    def coerce_list(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("campaign_id")
    @classmethod
    def validate_campaign_id(cls, v):
        if v is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", v):
            raise ValueError("campaign_id must be 1-128 URL-safe characters")
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
    next_cursor: Optional[str] = None
    campaign_id: Optional[str] = None
    # Which provider actually served this search, and which ones were tried and
    # passed over. Without it a fallback is invisible: the caller cannot tell
    # whether the leads came from the provider it expects, or why the shape of
    # the data changed between two identical-looking searches.
    provider: Optional[str] = None
    provider_attempts: Optional[list] = None


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

    await resolve_company_domain(params)
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
                # Nothing droppable is left. If Wiza is objecting to a filter
                # that defines the ICP, say which one instead of failing
                # anonymously — and never retry without it.
                blamed = [k for k in ICP_DEFINING if k in filters and k in err_text]
                if blamed:
                    raise HTTPException(
                        status_code=422,
                        detail=(f"Wiza rejected {', '.join(blamed)}, which defines this "
                                "search. Dropping it would return leads outside the "
                                "requested profile, so the search was not broadened."),
                    )
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

    await resolve_company_domain(params)
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
                # Same rule as the list workflow: an ICP-defining filter is
                # never dropped to make a request succeed.
                blamed = [k for k in ICP_DEFINING if k in filters and k in err_text]
                if blamed:
                    raise HTTPException(
                        status_code=422,
                        detail=(f"Wiza rejected {', '.join(blamed)}, which defines this "
                                "search. Dropping it would return leads outside the "
                                "requested profile, so the search was not broadened."),
                    )
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


async def resolve_company_domain(params: dict) -> None:
    """Turn a company_domain into a concrete `company` name for people search.

    Wiza's prospect filters can't target a company by domain, so a domain-only
    search is resolved to the company's real name via company enrichment — the
    same domain->company step Wiza's own Company tab does — and that exact name
    drives the job_company filter. Falls back to the domain's root label if
    enrichment finds nothing. Mutates `params` in place. Runs only on cache
    miss (from inside the Wiza fetchers), so it costs at most 2 credits per
    unique search, not per request.
    """
    domain = params.get("company_domain")
    company = params.get("company")
    # The ICP parser / frontend often puts the bare domain in the `company`
    # field too (e.g. company="workflows.io"). A domain isn't a usable company
    # name, so treat it as the domain to resolve rather than an already-resolved
    # company — otherwise we'd skip enrichment and text-match the raw domain,
    # which matches none of the company's actual employees.
    if company and looks_like_domain(company):
        domain = domain or company
        company = None
        params["company"] = None
    if not domain or company:
        return
    name = None
    try:
        enriched = await wiza_company_enrich({"company_domain": domain})
        if enriched:
            name = enriched.get("company_name")
    except HTTPException as exc:
        print(f"Domain resolve: company enrich failed for {domain} ({exc.detail})")
    if name:
        # Exact company entity — the same domain->company step Wiza's Company tab
        # runs. Use the canonical name alone and drop the domain so
        # build_wiza_filters doesn't also OR-in the root-label form, which pulls
        # in unrelated firms that merely share the word (e.g. "workflows").
        params["company"] = name
        params["company_domain"] = None
        print(f"Domain resolve: {domain} -> company '{name}' (exact)")
    else:
        # Enrichment found nothing — keep the domain so build_wiza_filters can
        # fall back to the root-label / full-domain job_company forms.
        params["company_domain"] = domain
        print(f"Domain resolve: {domain} -> no company match, using domain forms")


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
# Bytemine provider  (first in the chain — masked search, reveal on unlock)
# =============================================================================

# Bytemine's seniority vocabulary. "C-Team" is the canonical spelling; the API
# also accepts C-Level/C-Suite/CXO aliases, but sending the canonical value
# keeps what we send and what we log identical.
_BM_SENIORITY = {
    "founder": "Owner", "owner": "Owner", "partner": "Partner",
    "cxo": "C-Team", "c-level": "C-Team", "c-suite": "C-Team", "c_suite": "C-Team",
    "executive": "C-Team", "chief": "C-Team",
    "vp": "VP", "vice president": "VP",
    "director": "Director", "head": "Director",
    "manager": "Manager", "senior": "Senior",
    "entry": "Entry", "junior": "Entry", "intern": "Training", "training": "Training",
}

# Bytemine's employee bands. Note the top band is "10000+", not the "10001+"
# Wiza uses — sending the wrong string filters nothing.
_BM_EMPLOYEE_BANDS = [
    (1, 10, "1-10"), (11, 50, "11-50"), (51, 200, "51-200"),
    (201, 500, "201-500"), (501, 1000, "501-1000"),
    (1001, 5000, "1001-5000"), (5001, 10000, "5001-10000"),
]
_BM_TOP_BAND = "10000+"

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


def bytemine_employee_band(size: str):
    """Map a headcount range onto a Bytemine employee band, or None.

    Accepts the "N-M" / "N+" shapes the rest of the pipeline uses and picks the
    band containing the range's floor. Returns None when there is no number to
    read, so the caller can decline the search rather than drop the filter.
    """
    if not size:
        return None
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", str(size))]
    if not numbers:
        return None
    floor = min(numbers)
    for low, high, band in _BM_EMPLOYEE_BANDS:
        if floor <= high:
            return band
    return _BM_TOP_BAND


def build_bytemine_filters(p: dict) -> dict:
    """Translate internal search params into a Bytemine /contacts/search body.

    Raises ProviderUnsupported for anything Bytemine's contact search cannot
    express — notably country-level location and free-text keywords, which it
    has no field for. Crustdata handles both, so those searches fall through.
    """
    body: dict = {}

    if p.get("job_title"):
        body["jobTitles"] = [p["job_title"]]
    if p.get("seniority"):
        mapped = _BM_SENIORITY.get(str(p["seniority"]).strip().lower())
        if mapped:
            body["seniorityLevels"] = [mapped]
    if p.get("departments"):
        depts = p["departments"]
        body["departments"] = depts if isinstance(depts, list) else [depts]
    if p.get("industry"):
        body["industries"] = [p["industry"]]
    if p.get("company"):
        body["companyNames"] = [p["company"]]

    # Domains from an explicit company_domain and from a keyword resolved
    # against the company graph (see bytemine_resolve_keywords) are the same
    # constraint to /contacts/search, so they share one field.
    urls = list(p.get("company_domains") or [])
    if p.get("company_domain"):
        urls.insert(0, p["company_domain"])
    if urls:
        body["urls"] = list(dict.fromkeys(urls))

    if p.get("company_size"):
        band = bytemine_employee_band(p["company_size"])
        if not band:
            raise ProviderUnsupported("company_size", p["company_size"])
        body["employeeSizes"] = [band]

    # /contacts/search filters location by US state or city only. A country —
    # which is what the ICP parser usually produces — has no field, so rather
    # than search the whole world under a country filter the user set, hand the
    # query to a provider that supports it.
    location = p.get("location") or p.get("company_location")
    if location:
        token = str(location).strip()
        if token.upper() in _US_STATES:
            body["states"] = [token.upper()]
        elif len(token) > 2 and token.upper() not in {"US", "USA", "GB", "UK"}:
            body["cities"] = [token]
        else:
            raise ProviderUnsupported("location", location)

    # /contacts/search has no free-text field. Keywords are meant to be resolved
    # to company domains first — bytemine_resolve_keywords does that and clears
    # the key. Reaching here with one still set means an unresolved term would
    # be silently dropped, so refuse rather than run a broader search.
    if p.get("keywords"):
        raise ProviderUnsupported("keywords", p["keywords"])

    if not body:
        raise HTTPException(status_code=400, detail="At least one search parameter required")
    return body


async def bytemine_call(path: str, body: dict, timeout: float = 60.0) -> dict:
    """POST one Bytemine gateway request and return the decoded payload.

    Every endpoint goes through a single URL with the real path and method in
    the JSON body, so this is the only place that shape is constructed.
    """
    headers = {
        "x-amz-security-token": settings.bytemine_api_key or "",
        "Content-Type": "application/json",
    }
    envelope = {"path": path, "method": "POST", "body": body}
    print(f"Bytemine {path} body: {json.dumps(body)[:400]}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(BYTEMINE_GATEWAY, headers=headers, json=envelope)
        print(f"Bytemine {path} status: {resp.status_code} {resp.text[:300]}")
        if resp.status_code == 402:
            raise HTTPException(
                status_code=402,
                detail="Bytemine credits exhausted — top up the Bytemine account to keep searching",
            )
        if resp.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Bytemine rate limit reached — please try again in a moment",
            )
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Bytemine {path} error: {resp.text[:300]}",
            )
        return resp.json()


# How many companies a keyword is allowed to resolve to. /b2b-search bills one
# credit per company returned, so this is a real cost per keyword search; it is
# also the ceiling on how many accounts the contact search can then draw from.
BYTEMINE_KEYWORD_COMPANIES = 50


async def bytemine_resolve_keywords(params: dict) -> dict:
    """Turn a free-text term into the company domains it describes.

    /contacts/search has no free-text field, which is why keywords used to be
    refused outright. /b2b-search does: `keywords` matches company descriptions,
    which is where a segment like "AI SaaS" is actually written down. Resolving
    the term to companies first and then filtering contacts to those domains
    keeps it a hard constraint instead of dropping it.

    Returns params unchanged when there is nothing to resolve. On a resolved
    keyword the key is replaced by `company_domains`; an empty list there means
    the term genuinely matched no company, which is a real no-results answer
    rather than a reason to search without it.
    """
    keywords = (params.get("keywords") or "").strip()
    if not keywords:
        return params

    body: dict = {
        "keywords": keywords,
        "page": 1,
        "page_size": BYTEMINE_KEYWORD_COMPANIES,
        # A contact search filters by domain, so a company without one cannot
        # be used and would only consume credits.
        "has_website": True,
    }
    if params.get("industry"):
        body["industry"] = params["industry"]
    if params.get("company_size"):
        lo, hi = _size_bounds(params["company_size"])
        if lo is not None:
            body["min_employees"] = lo
        if hi is not None:
            body["max_employees"] = hi

    location = params.get("company_location") or params.get("location")
    if location:
        token = str(location).strip()
        if token.upper() in _US_STATES:
            body["state"] = token.upper()
        elif len(token) == 2:
            body["country"] = token.upper()
        elif token.lower() in _LOCATION_ALIASES:
            body["country"] = "US" if _LOCATION_ALIASES[token.lower()] == "United States" else "GB"
        else:
            body["city"] = token

    data = await bytemine_call("/b2b-search", body)
    companies = data.get("data") or data.get("results") or []
    domains = []
    for company in companies:
        host = domain_host(company.get("website") or company.get("domain") or "")
        if host and host not in domains:
            domains.append(host)

    print(f"Bytemine keyword {keywords!r} resolved to {len(domains)} company domains")
    resolved = {k: v for k, v in params.items() if k != "keywords"}
    resolved["company_domains"] = domains
    return resolved


async def bytemine_person_search(params: dict, limit: int, cursor: str = None,
                                 offset: int = 0) -> dict:
    """Search Bytemine prospects. Results come back masked; /enrich unlocks them.

    Bytemine pages by 0-indexed page number, so an offset is converted rather
    than ignored — returning page 1 for every page is what made every search
    look like it had the same handful of leads.
    """
    # Check the rest of the request is expressible before resolving keywords:
    # /b2b-search bills per company returned, and spending that on a search
    # about to be refused for an unrelated filter is money for nothing. A
    # keyword on its own is a complete search once resolved, so there is nothing
    # to pre-check and an empty body here would be the wrong error.
    others = {k: v for k, v in params.items() if k != "keywords"}
    if params.get("keywords") and any(
            others.get(k) for k in ("job_title", "seniority", "departments",
                                    "industry", "company", "company_domain",
                                    "company_size", "location", "company_location")):
        build_bytemine_filters(others)

    params = await bytemine_resolve_keywords(params)
    # The keyword described companies and none exist; a contact search without
    # it would answer a different question.
    if params.get("company_domains") == []:
        return {"profiles": [], "total": 0, "next_cursor": None, "credits_used": None}

    body = build_bytemine_filters(params)
    size = max(min(limit, 100), 1)
    body["pageSize"] = size
    body["page"] = max(offset, 0) // size
    if cursor:
        body["after"] = cursor

    data = await bytemine_call("/contacts/search", body)
    profiles = data.get("data") or []
    pagination = data.get("pagination") or {}
    total = pagination.get("total")
    return {
        "profiles": profiles,
        "total": len(profiles) if total is None else total,
        "next_cursor": pagination.get("after") or data.get("after"),
        "credits_used": data.get("credits_used"),
    }


async def fetch_from_bytemine(params: dict) -> list:
    """People-search fetcher for /search — returns raw Bytemine profiles."""
    limit = max(min(params.get("limit", 10), 100), 1)
    data = await bytemine_person_search(params, limit)
    return data["profiles"]


def _bm_masked(value) -> bool:
    """True when a masked field indicates the real value exists behind it.

    Search returns "***" where a contact has an email or phone, so presence of
    the mask — not its content — is the availability signal.
    """
    return bool(value) and str(value).strip() not in {"", "null", "None"}


def transform_bytemine_profile(profile: dict, search_params: dict = None) -> dict:
    """Transform a Bytemine contact into the internal lead shape.

    Search returns email and phone masked as "***"; the real values come from
    /contacts/unlock keyed on `pid`, so they are None here and `pid` is carried
    through for the reveal.
    """
    search_params = search_params or {}
    first = profile.get("first_name")
    last = profile.get("last_name")
    full = " ".join(x for x in (first, last) if x) or profile.get("full_name")

    city, state = profile.get("city"), profile.get("state")
    location = ", ".join(x for x in (city, state) if x and not isinstance(x, int)) or None

    return {
        "contact_name": full,
        "first_name": first,
        "last_name": last,
        "job_title": profile.get("job_title"),
        "linkedin_url": profile.get("linkedin_url"),
        # The unlock identifier. /enrich needs it to reveal this exact record.
        "bytemine_pid": profile.get("pid"),
        "business_email": None,
        "email_status": "available" if _bm_masked(profile.get("email")) else None,
        "email_available": _bm_masked(profile.get("email")),
        "phone": None,
        "phone_available": _bm_masked(profile.get("phone")),
        "location": location,
        "company_name": profile.get("company_name"),
        "company_domain": profile.get("company_domain"),
        "industry": profile.get("company_industry") or search_params.get("industry"),
        "company_size": (profile.get("company_employee_range")
                         or search_params.get("company_size")),
        "company_headcount": None,
        "company_revenue": profile.get("company_revenue_range"),
        "company_funding": None,
        "technologies": search_params.get("technologies"),
        "department": profile.get("department"),
        "company_linkedin": None,
        "profile_picture": None,
        "raw_data": profile,
    }


def transform_bytemine_unlocked(record: dict) -> dict:
    """Transform an unlocked/enriched Bytemine contact into the internal shape.

    Unlock and enrich return the real email and phone, plus the same
    firmographics search returns under slightly different keys.
    """
    lead = transform_bytemine_profile(record)
    email = (record.get("work_email") or record.get("email")
             or record.get("personal_email"))
    phone = (record.get("phone") or record.get("direct_dial")
             or record.get("mobile_phone"))
    lead.update({
        "contact_name": record.get("full_name") or lead["contact_name"],
        "business_email": email if email and "*" not in str(email) else None,
        "email_status": "verified" if email and "*" not in str(email) else "no_email",
        "email_available": bool(email and "*" not in str(email)),
        "phone": phone if phone and "*" not in str(phone) else None,
        "phone_available": bool(phone and "*" not in str(phone)),
        "linkedin_url": record.get("linkedin_profile") or lead["linkedin_url"],
        "company_linkedin": record.get("company_linkedin_profile"),
    })
    return lead


async def bytemine_unlock(pid: str) -> dict:
    """Reveal one contact by PID via /contacts/unlock. Costs 1 credit."""
    data = await bytemine_call("/contacts/unlock", {"pids": [str(pid)]})
    records = data.get("data") or data.get("results") or []
    if isinstance(records, dict):
        records = [records]
    return records[0] if records else {}


async def bytemine_enrich(identifiers: dict) -> dict:
    """Reveal a contact from an email, phone, LinkedIn URL or name + domain."""
    body = {k: v for k, v in identifiers.items() if v}
    if not body:
        return {}
    # At least one match filter must be true or the upstream rejects the call.
    body.setdefault("hasWorkEmail", True)
    body.setdefault("hasPhone", True)
    data = await bytemine_call("/contacts/enrich", body)
    record = data.get("data") or {}
    if isinstance(record, list):
        record = record[0] if record else {}
    return record


# =============================================================================
# Crustdata provider  (fallback — people API can filter by company domain)
# =============================================================================

# Crustdata person-search field paths (v2025-11-01).
_CD_DOMAIN = "experience.employment_details.current.company_website_domain"
_CD_COMPANY = "experience.employment_details.current.company_name"
_CD_TITLE = "experience.employment_details.current.title"
_CD_INDUSTRY = "experience.employment_details.current.company_industries"
_CD_HEADCOUNT = "experience.employment_details.current.company_headcount_latest"
_CD_LOCATION = "basic_profile.location.full_location"
_CD_COUNTRY = "basic_profile.location.country"
_CD_SENIORITY = "experience.employment_details.current.seniority_level"
_CD_COMPANY_LOCATION = "experience.employment_details.current.company_hq_location"

_CD_COUNTRY_ALIASES = {
    "us": "United States", "u.s": "United States", "u.s.": "United States",
    "usa": "United States", "united states": "United States",
    "uk": "United Kingdom", "u.k": "United Kingdom", "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
}

_CD_SENIORITY_MAP = {
    "entry": "Entry", "training": "Training", "intern": "Entry",
    "junior": "Entry", "senior": "Senior", "manager": "Manager",
    "head": "Manager", "director": "Director", "partner": "Partner",
    "vp": "VP", "c_suite": "CXO", "cxo": "CXO", "owner": "Owner",
    "founder": "Owner",
}


def _size_bounds(size: str):
    """Parse a headcount range label ('11-50', '10001+') into (min, max)."""
    s = (size or "").replace(",", "").strip()
    if s.endswith("+"):
        try:
            return int(s[:-1]), None
        except ValueError:
            return None, None
    if "-" in s:
        lo, _, hi = s.partition("-")
        try:
            return int(lo), int(hi)
        except ValueError:
            return None, None
    return None, None


def build_crustdata_filters(p: dict):
    """Translate internal search params into a Crustdata person/search filter.

    Returns a single condition, an {op:"and", conditions:[...]} group, or None
    if there's nothing to filter on. The key win over Wiza: a domain filters
    people by their company's exact website domain, not a fuzzy name match.
    """
    conds: list = []

    company = p.get("company")
    domain = p.get("company_domain")
    # A `company` value that is really a bare domain is treated as the domain.
    if company and not domain and looks_like_domain(company):
        domain, company = company, None
    if domain:
        conds.append({"field": _CD_DOMAIN, "type": "=", "value": domain_host(domain)})
    elif company:
        # Fuzzy, typo-tolerant match on the company name when we have no domain.
        conds.append({"field": _CD_COMPANY, "type": "(.)", "value": company})

    if p.get("job_title"):
        conds.append({"field": _CD_TITLE, "type": "(.)", "value": p["job_title"]})

    if p.get("seniority"):
        raw_seniority = p["seniority"].strip()
        seniority = _CD_SENIORITY_MAP.get(raw_seniority.lower(), raw_seniority)
        conds.append({"field": _CD_SENIORITY, "type": "=", "value": seniority})

    if p.get("location"):
        raw_location = p["location"].strip()
        country = _CD_COUNTRY_ALIASES.get(raw_location.lower())
        if country:
            conds.append({"field": _CD_COUNTRY, "type": "=", "value": country})
        else:
            conds.append({"field": _CD_LOCATION, "type": "(.)", "value": raw_location})

    if p.get("company_location"):
        conds.append({
            "field": _CD_COMPANY_LOCATION,
            "type": "(.)",
            "value": p["company_location"].strip(),
        })

    if p.get("industry"):
        conds.append({"field": _CD_INDUSTRY, "type": "(.)", "value": p["industry"]})

    if p.get("company_size"):
        lo, hi = _size_bounds(p["company_size"])
        if lo is not None:
            conds.append({"field": _CD_HEADCOUNT, "type": "=>", "value": lo})
        if hi is not None:
            conds.append({"field": _CD_HEADCOUNT, "type": "=<", "value": hi})

    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"op": "and", "conditions": conds}


async def crustdata_person_search(
    params: dict,
    limit: int,
    cursor: str = None,
    exclude_profiles: list[str] = None,
) -> dict:
    """Call Crustdata POST /person/search with stable cursor pagination."""
    filters = build_crustdata_filters(params)
    semantic_query = (params.get("keywords") or "").strip()
    if not filters and not semantic_query:
        raise HTTPException(status_code=400, detail="At least one search parameter required")

    body = {
        "limit": max(min(limit, 100), 1),
        # social_handles carries the person's LinkedIn URL — the identifier the
        # enrich endpoint needs to reveal their work email/phone on demand.
        "fields": ["crustdata_person_id", "basic_profile", "experience", "contact", "social_handles"],
    }
    if not semantic_query:
        # A deterministic unique key prevents page drift while walking cursors.
        # Crustdata rejects the whole request with 400 "sorts are not supported
        # when using semantic search" if this is sent alongside a search query,
        # so a keyword search orders by relevance and paginates on the cursor
        # alone.
        body["sorts"] = [{"field": "crustdata_person_id", "order": "asc"}]
    if filters:
        body["filters"] = filters
    if semantic_query:
        body["search"] = {"query": semantic_query, "mode": "hybrid"}
        # Keep structured filters as hard constraints; keywords only rank within
        # the requested ICP rather than broadening it.
        if filters:
            body["mode"] = "exact"
    if cursor:
        body["cursor"] = cursor
    exclusions = list(dict.fromkeys(x for x in (exclude_profiles or []) if x))
    if exclusions:
        body["post_processing"] = {"exclude_profiles": exclusions}
    headers = {
        "Authorization": f"Bearer {settings.crustdata_api_key}",
        "x-api-version": CRUSTDATA_VERSION,
        "Content-Type": "application/json",
    }
    print(f"Crustdata person search body: {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{CRUSTDATA_BASE}/person/search", headers=headers, json=body,
        )
        print(f"Crustdata search status: {resp.status_code} {resp.text[:300]}")
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Crustdata rate limit reached — please try again in a moment")
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail=f"Crustdata search error: {resp.text[:300]}")

        data = resp.json()
        profiles = data.get("profiles") or []
        total = data.get("total_count")
        return {
            "profiles": profiles,
            "total": len(profiles) if total is None else total,
            "next_cursor": data.get("next_cursor"),
        }


async def fetch_from_crustdata(params: dict) -> list:
    """People-search fetcher for /search — returns raw Crustdata profiles."""
    limit = max(min(params.get("limit", 10), 100), 1)
    data = await crustdata_person_search(params, limit)
    return data["profiles"]


def _cd_current(profile: dict) -> dict:
    """The person's primary current employment record (or {})."""
    cur = (((profile.get("experience") or {}).get("employment_details") or {}).get("current")) or []
    return cur[0] if cur else {}


def transform_crustdata_profile(profile: dict, search_params: dict = None) -> dict:
    """Transform a Crustdata person into the internal lead shape.

    Search returns email/phone *availability* flags (has_business_email, …); the
    actual values are pulled on demand by /enrich, so email/phone are None here.
    """
    search_params = search_params or {}
    bp = profile.get("basic_profile") or {}
    cur = _cd_current(profile)
    contact = profile.get("contact") or {}
    loc = bp.get("location") or {}

    name = bp.get("name")
    first, last = bp.get("first_name"), bp.get("last_name")
    if name and not (first or last):
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0], " ".join(parts[1:])
        elif parts:
            first = parts[0]

    industries = cur.get("company_industries") or []
    website = cur.get("company_website") or ""
    domain = domain_host(website) if website else None
    linkedin_url = (((profile.get("social_handles") or {})
                     .get("professional_network_identifier") or {}).get("profile_url"))

    return {
        "contact_name": name,
        "first_name": first,
        "last_name": last,
        "job_title": cur.get("title") or bp.get("current_title"),
        # LinkedIn URL is the identifier /enrich uses to reveal email/phone.
        "linkedin_url": linkedin_url,
        "business_email": None,
        "email_status": "available" if contact.get("has_business_email") else None,
        "email_available": bool(contact.get("has_business_email")),
        "phone": None,
        "phone_available": bool(contact.get("has_phone_number")),
        "location": loc.get("raw"),
        "company_name": cur.get("name"),
        "company_domain": domain,
        "industry": (cur.get("company_professional_network_industry")
                     or (industries[0] if industries else None)
                     or search_params.get("industry")),
        "company_size": cur.get("company_headcount_range") or search_params.get("company_size"),
        "company_headcount": cur.get("company_headcount_latest"),
        "company_revenue": None,
        "company_funding": None,
        "technologies": search_params.get("technologies"),
        "department": (bp.get("normalized_title") or {}).get("department"),
        "company_linkedin": cur.get("company_professional_network_profile_url"),
        "profile_picture": bp.get("profile_picture_permalink"),
        "raw_data": profile,
    }


async def crustdata_person_enrich(linkedin_url: str) -> dict:
    """Reveal a person's work email + phone via Crustdata POST /person/enrich.

    Crustdata's enrich only accepts a LinkedIn profile URL as the identifier,
    and email/phone are additive fields that must be requested explicitly (they
    cost extra credits — the on-demand reveal, same idea as Wiza's reveal).
    Returns the matched person_data object, or {} if no match.
    """
    body = {
        "professional_network_profile_urls": [linkedin_url],
        "fields": ["basic_profile", "experience", "contact", "social_handles"],
    }
    headers = {
        "Authorization": f"Bearer {settings.crustdata_api_key}",
        "x-api-version": CRUSTDATA_VERSION,
        "Content-Type": "application/json",
    }
    print(f"Crustdata enrich request: {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{CRUSTDATA_BASE}/person/enrich", headers=headers, json=body,
        )
        print(f"Crustdata enrich status: {resp.status_code} {resp.text[:200]}")
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail="Crustdata rate limit — try again in a moment")
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail=f"Crustdata enrich error: {resp.text[:300]}")

        data = resp.json()
        # Response shape: [{matched_on, matches: [{person_data, confidence_score}]}]
        if isinstance(data, list) and data:
            matches = data[0].get("matches") or []
            if matches:
                return matches[0].get("person_data") or {}
        return {}


def transform_crustdata_enrich(person_data: dict) -> dict:
    """Transform a Crustdata enrich person_data object into the internal lead."""
    lead = transform_crustdata_profile(person_data)
    contact = person_data.get("contact") or {}

    biz = contact.get("business_emails") or []
    primary = biz[0] if biz else None
    if not primary:
        personal = contact.get("personal_emails") or []
        primary = personal[0] if personal else None
    phones = contact.get("phone_numbers") or []

    lead["business_email"] = primary.get("email") if primary else None
    lead["email_status"] = primary.get("status") if primary else None
    lead["email_available"] = bool(biz)
    lead["phone"] = (phones[0].get("number") or phones[0].get("phone_number")
                     or (phones[0] if isinstance(phones[0], str) else None)) if phones else None
    lead["phone_available"] = bool(phones)
    return lead


def aggregate_companies_crustdata(profiles: list) -> list:
    """Roll Crustdata people up into unique companies with sample contacts.

    Crustdata already returns full firmographics per person, so each company is
    populated directly from the profiles — no separate enrichment call needed.
    """
    companies: dict = {}
    for p in profiles:
        bp = p.get("basic_profile") or {}
        cur = _cd_current(p)
        name = cur.get("name")
        website = cur.get("company_website") or ""
        domain = domain_host(website) if website else None
        key = (domain or name or "").strip().lower()
        if not key:
            continue
        if key not in companies:
            industries = cur.get("company_industries") or []
            companies[key] = {
                "company_name": name,
                "company_domain": domain,
                "industry": cur.get("company_professional_network_industry")
                            or (industries[0] if industries else None),
                "company_size": cur.get("company_headcount_range"),
                "employee_count": cur.get("company_headcount_latest"),
                "location": cur.get("company_hq_location"),
                "country": cur.get("company_headquarters_country"),
                "company_type": cur.get("company_type"),
                "linkedin": cur.get("company_professional_network_profile_url"),
                "logo": cur.get("company_profile_picture_permalink"),
                "enriched": True,
                "matched_contacts": 0,
                "sample_contacts": [],
            }
        entry = companies[key]
        entry["matched_contacts"] += 1
        if len(entry["sample_contacts"]) < 5:
            entry["sample_contacts"].append({
                "contact_name": bp.get("name"),
                "job_title": cur.get("title") or bp.get("current_title"),
                "linkedin_url": None,
            })
    return sorted(companies.values(), key=lambda c: c["matched_contacts"], reverse=True)


# =============================================================================
# Cache helper (generic, cache-first for any endpoint)
# =============================================================================

def _profile_url(profile: dict) -> Optional[str]:
    return (((profile.get("social_handles") or {})
             .get("professional_network_identifier") or {}).get("profile_url"))


def _profile_lead_key(profile: dict) -> Optional[str]:
    person_id = profile.get("crustdata_person_id")
    if person_id is not None:
        return f"id:{person_id}"
    profile_url = _profile_url(profile)
    if profile_url:
        return f"url:{hashlib.sha256(profile_url.encode()).hexdigest()}"
    return None


def dedupe_crustdata_profiles(profiles: list[dict]) -> list[dict]:
    """Remove duplicate people within a provider page while preserving order."""
    output, seen = [], set()
    for profile in profiles:
        key = _profile_lead_key(profile)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        output.append(profile)
    return output


async def campaign_seen(campaign_id: str) -> tuple[set[str], list[str]]:
    """Return the campaign's durable lead keys and usable provider exclusions."""
    try:
        async with async_session() as session:
            rows = (await session.execute(
                select(CampaignSeenLead).where(CampaignSeenLead.campaign_id == campaign_id)
            )).scalars().all()
            return {row.lead_key for row in rows}, [row.profile_url for row in rows if row.profile_url]
    except Exception as e:
        print(f"WARNING: campaign seen lookup failed ({e})")
        return set(), []


async def record_new_campaign_profiles(campaign_id: str, profiles: list[dict]) -> list[dict]:
    """Atomically claim unseen profiles and return only rows claimed by this call.

    PostgreSQL's ON CONFLICT guard makes concurrent requests for one campaign
    safe: only one request can claim a given person and surface it to the user.
    Profiles without a stable provider ID or profile URL are returned but cannot
    be tracked durably.
    """
    keyed = [(profile, _profile_lead_key(profile)) for profile in profiles]
    trackable = [(profile, key) for profile, key in keyed if key]
    untrackable = [profile for profile, key in keyed if not key]
    if not trackable:
        return untrackable

    values = [{
        "campaign_id": campaign_id,
        "lead_key": key,
        "crustdata_person_id": (str(profile.get("crustdata_person_id"))
                                 if profile.get("crustdata_person_id") is not None else None),
        "profile_url": _profile_url(profile),
        "created_at": datetime.utcnow(),
    } for profile, key in trackable]

    try:
        async with async_session() as session:
            stmt = (pg_insert(CampaignSeenLead)
                    .values(values)
                    .on_conflict_do_nothing(index_elements=["campaign_id", "lead_key"])
                    .returning(CampaignSeenLead.lead_key))
            inserted = set((await session.execute(stmt)).scalars().all())
            await session.commit()
        return [profile for profile, key in keyed if key is None or key in inserted]
    except Exception as e:
        # Availability wins if the seen-ledger DB is temporarily unavailable;
        # the provider-level exclusions still prevent most repeats.
        print(f"WARNING: campaign seen store failed ({e})")
        return profiles

async def cache_lookup(search_hash: str):
    """Return the CachedSearch row for a hash, or None. Never raises: a DB outage
    degrades to a live (uncached) fetch instead of failing the whole request."""
    try:
        async with async_session() as session:
            stmt = select(CachedSearch).where(CachedSearch.search_hash == search_hash)
            cached = (await session.execute(stmt)).scalar_one_or_none()
            if not cached:
                return None
            ttl = max(settings.search_cache_ttl_seconds, 0)
            if ttl == 0:
                return None
            cached_at = cached.updated_at or cached.created_at
            if cached_at and datetime.utcnow() - cached_at > timedelta(seconds=ttl):
                print(f"Cache EXPIRED for hash: {search_hash}")
                return None
            return cached
    except Exception as e:
        print(f"WARNING: cache lookup failed ({e}) — proceeding without cache")
        return None


async def cache_store(search_hash: str, params: dict, raw: list) -> None:
    """Persist a search result; best-effort. Never raises — if the DB is down we
    simply don't cache, rather than failing a search that already succeeded."""
    try:
        async with async_session() as session:
            existing = (await session.execute(
                select(CachedSearch).where(CachedSearch.search_hash == search_hash)
            )).scalar_one_or_none()
            if existing:
                existing.search_params = json.dumps(params)
                existing.results = json.dumps(raw)
                existing.updated_at = datetime.utcnow()
            else:
                session.add(CachedSearch(
                    search_hash=search_hash,
                    search_params=json.dumps(params),
                    results=json.dumps(raw),
                ))
            await session.commit()
    except Exception as e:
        print(f"WARNING: cache store failed ({e}) — result not cached")


async def cached_or_fetch(kind: str, params: dict, fetcher) -> tuple[list, bool]:
    """Cache-first wrapper. `fetcher` is an async callable returning a LIST of
    raw dicts (always a list so /debug and /cache/* stay consistent). Returns
    (raw_results, from_cache). The cache key is namespaced by `kind`.

    Caching is best-effort: if the database is unreachable the search still runs
    against Wiza and returns live results (just uncached).
    """
    hashable = {"_kind": kind, **params}
    search_hash = generate_search_hash(hashable)
    print(f"[{kind}] search hash: {search_hash}")

    cached = await cache_lookup(search_hash)
    if cached:
        print(f"[{kind}] cache HIT")
        return json.loads(cached.results), True

    print(f"[{kind}] cache MISS — calling Wiza")
    raw = await fetcher()
    if not isinstance(raw, list):
        raw = [raw] if raw else []

    await cache_store(search_hash, hashable, raw)
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
    prov, degraded = provider_state()
    if degraded:
        print("WARNING: SEARCH_PROVIDER=crustdata but CRUSTDATA_API_KEY is unset.")
        print("Falling back to Wiza's credit-free preview search: results are capped")
        print("at 30 per page and carry no firmographics beyond company/industry.")
        print("Set CRUSTDATA_API_KEY to restore full search.")
    else:
        print(f"Search provider: {prov}")

    try:
        await init_db()
    except Exception as e:
        print(f"WARNING: Database initialization failed: {e}")
        print("App will continue — DB retried on first request.")


@app.get("/health")
async def health_check():
    """Liveness plus the provider chain actually in use.

    The chain is reported because a missing key silently changes which upstream
    serves every search, and each returns a different shape. Without this the
    change is invisible until someone reads the logs or notices the results.
    """
    prov, degraded = provider_state()
    chain = provider_chain()
    preferred = (settings.search_provider or PROVIDER_ORDER[0]).strip().lower()
    return {
        "status": "healthy",
        "search_provider": prov,
        "provider_chain": chain,
        "degraded": degraded,
        "degraded_reason": (
            f"SEARCH_PROVIDER={preferred} but no API key is configured for it; "
            f"searches are served by {prov}"
        ) if degraded else None,
    }


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
    # Bytemine's unlock identifier, carried on every lead its search returns.
    # Unlocking by PID reveals the exact record that was shown, so it cannot
    # resolve to a different person the way a name lookup can.
    bytemine_pid: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("bytemine_pid", "pid"))

    model_config = {"populate_by_name": True}


@app.post("/enrich")
async def enrich_lead(request: EnrichRequest):
    """
    Enrich a single lead, trying each configured provider in turn.

    Each provider reveals from the identifier its own search hands back, so the
    order follows what the lead is carrying rather than a fixed preference: a
    Bytemine PID unlocks the exact record that was shown, a LinkedIn URL goes to
    Crustdata, and anything left (email-only, or name + company) falls through to
    the Wiza reveal, which accepts the widest set of identifiers.
    """
    chain = provider_chain()

    # Bytemine reveal: unlock by PID when the lead came from its search,
    # otherwise match on whatever identifier we do have.
    if "bytemine" in chain:
        record = {}
        try:
            if request.bytemine_pid:
                record = await bytemine_unlock(request.bytemine_pid)
            elif request.linkedin_url or request.email:
                record = await bytemine_enrich({
                    "linkedin": request.linkedin_url,
                    "email": request.email,
                    "firstName": (request.full_name or "").split(" ")[0] or None,
                    "lastName": " ".join((request.full_name or "").split(" ")[1:]) or None,
                    "companyDomain": request.company_domain,
                    "companyName": request.company,
                })
        except HTTPException as exc:
            # A reveal that fails at one provider should still be attempted at
            # the next — the lead is the same person either way.
            print(f"Bytemine reveal failed ({exc.status_code}); trying next provider")
            record = {}
        if record:
            lead = transform_bytemine_unlocked(record)
            return {
                "success": True,
                "provider": "bytemine",
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "lead": lead,
            }

    # Crustdata reveal path: fast, single call, keyed on the LinkedIn URL that
    # Crustdata-sourced leads carry.
    if "crustdata" in chain and request.linkedin_url:
        try:
            person = await crustdata_person_enrich(request.linkedin_url)
        except HTTPException as exc:
            print(f"Crustdata reveal failed ({exc.status_code}); falling through to Wiza")
            person = {}
        if person:
            lead = transform_crustdata_enrich(person)
            return {
                "success": True,
                "provider": "crustdata",
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "lead": lead,
            }
        # No Crustdata match — fall through to Wiza with whatever identifiers we have.

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
                    "provider": "wiza",
                    "enrichment_status": "complete" if data.get("status") != "failed" else "failed",
                    "lead": transform_reveal_contact(contact),
                }

        raise HTTPException(status_code=504, detail="Wiza enrichment timed out")


async def _llm_parse_icp(text: str) -> dict:
    """Parse an ICP via the Anthropic API. Raises on any API/parse error so the
    caller can fall back to the rule-based parser. Requires anthropic_api_key.
    """
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
  "industry": "lowercase industry from Wiza's vocabulary e.g. computer software, financial services, hospital & health care",
  "technologies": ["tech stack strings e.g. Salesforce, HubSpot, AWS"],
  "keywords": "additional search terms",
  "intent_topics": ["buyer intent — only from: Funding, IPO, Mergers, Investment"],
  "revenue_min": integer in USD,
  "revenue_max": integer in USD,
  "job_change_days": integer e.g. 90 if recently changed jobs is mentioned
}}

ICP Description: {request.text}

Return only the JSON object, no explanation, no markdown fences."""

    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(
        block.text for block in message.content
        if getattr(block, "type", None) == "text"
    ).strip()
    return extract_json_object(raw)


@app.post("/parse-icp")
async def parse_icp(request: ICPParseRequest):
    """Parse a plain-English ICP description into structured Wiza search filters.

    Rule-based by default — deterministic, instant, and free (no API credits).
    When USE_LLM_PARSER is enabled and an Anthropic key is set, the LLM also runs
    and fills any fields the rules didn't capture; if it errors (e.g. no credits)
    the rule-based result is used unchanged.
    """
    filters = heuristic_parse_icp(request.text)

    if settings.use_llm_parser and settings.anthropic_api_key:
        try:
            llm = await _llm_parse_icp(request.text)
            for k, v in llm.items():
                if v not in (None, "", []) and k not in filters:
                    filters[k] = v
        except Exception as e:
            print(f"ICP parser: LLM booster skipped, using rule-based result ({e})")

    # Drop a bare keyword fallback if the rules (or LLM) found real filters —
    # but never drop a term the taxonomy cannot express. For "AI SaaS" the
    # keyword *is* the query, and discarding it here left industry=computer
    # software as the entire search.
    if (len(filters) > 1 and filters.get("keywords") == request.text
            and not semantic_terms_in(request.text)):
        filters.pop("keywords", None)

    return {"success": True, "filters": filters}


# =============================================================================
# Search Endpoint
# =============================================================================

STRUCTURED_FIELDS = {
    "job_title", "departments", "seniority", "location", "company_location",
    "company", "company_domain", "company_size", "industry", "technologies",
    "keywords", "intent_topics", "revenue_min", "revenue_max", "job_change_days",
}

SEARCH_CONTROL_FIELDS = {
    "cursor", "campaign_id", "exclude_profiles", "refresh", "allow_broad_search",
}


async def resolve_search_params(request: SearchRequest) -> dict:
    """Turn a SearchRequest into concrete Wiza params.

    If only a natural-language `query` was provided, parse it into structured
    filters via the ICP parser (falling back to a keyword search). Shared by
    /search, /prospects/preview and /company/search.
    """
    params = {
        k: v for k, v in request.model_dump().items()
        if k not in SEARCH_CONTROL_FIELDS and v is not None and v != "" and v != []
    }
    print(f"=== SEARCH REQUEST ===\nFiltered params: {params}")

    raw_query = params.pop("query", None)

    # The frontend search box sends its text as `keywords`, not `query`. When
    # `keywords` is the only input, treat it as a natural-language query so domain
    # detection and ICP parsing run — otherwise a bare domain like "workflows.io"
    # stays a job_title keyword match and never reaches the company path.
    if not raw_query and params.get("keywords") and not any(
        params.get(fld) for fld in STRUCTURED_FIELDS if fld != "keywords"
    ):
        raw_query = params.pop("keywords")

    if raw_query and not any(params.get(f) for f in STRUCTURED_FIELDS):
        parsed_filters: dict = {}
        # The rule-based parser is free and deterministic — always run it. The
        # LLM booster inside parse_icp only fires when enabled + an Anthropic key
        # is set, so this is safe without one (and is the whole point of #5).
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
            # Parser gave us nothing (often because it's offline). A bare domain
            # in the query is unambiguous — route it to company_domain (which
            # build_wiza_filters turns into a job_company match) rather than
            # dumping the whole query into keywords -> job_title, which rarely
            # matches anything.
            domain = find_domain_in_text(raw_query)
            if domain:
                print(f"ICP parse yielded no filters — routing domain '{domain}' to company_domain")
                params["company_domain"] = domain
            else:
                print("ICP parse yielded no filters — using raw query as keyword search")
                params["keywords"] = raw_query

        print(f"Final params: {params}")

    return params


def enforce_crustdata_search_scope(params: dict, allow_broad_search: bool) -> None:
    """Reject the unsafe endpoint of the old progressive fallback.

    The caller may still intentionally perform a title-only search, but it must
    say so. This converts silent ICP broadening into a visible 422 response.
    """
    narrowing_fields = {
        "company", "company_domain", "location", "company_location",
        "company_size", "industry", "seniority", "keywords", "departments",
        "technologies", "intent_topics", "revenue_min", "revenue_max",
        "job_change_days",
    }
    if params.get("job_title") and not any(params.get(k) for k in narrowing_fields):
        if not allow_broad_search:
            raise HTTPException(
                status_code=422,
                detail=("Refusing a title-only Crustdata search because it can silently broaden "
                        "the requested ICP. Retain at least one narrowing filter or set "
                        "allow_broad_search=true for a deliberate title-only search."),
            )

    unsupported = [
        field for field in (
            "departments", "technologies", "intent_topics", "revenue_min",
            "revenue_max", "job_change_days", "signals",
        )
        if params.get(field)
    ]
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=("Crustdata does not map these requested filters yet: "
                    f"{', '.join(unsupported)}. Refusing to silently ignore them."),
        )


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
    chain = provider_chain()
    prov, degraded = provider_state()
    if request.cursor and prov not in ("crustdata", "bytemine"):
        raise HTTPException(
            status_code=422,
            detail="Cursor pagination is only available with Crustdata or Bytemine")

    # Namespace the cache by the whole chain: the same params can be served by
    # a different provider once a key is added or a balance runs out, and the
    # shapes differ. The stored payload records which provider produced it so a
    # cache hit is transformed the way it was written.
    # The degraded path returns preview-shaped profiles, not the enriched
    # contacts the full Wiza workflow yields, so it needs its own namespace.
    cache_params = {**params, "_provider": "wiza-preview" if degraded else "+".join(chain)}
    # Page is part of the identity of a result set. Leaving it out is what let
    # page 2 be answered from page 1's cached rows.
    if request.start_offset:
        cache_params["_offset"] = request.start_offset
    if request.cursor:
        cache_params["_cursor"] = request.cursor
    if request.exclude_profiles:
        cache_params["_exclude_profiles"] = sorted(set(request.exclude_profiles))
    search_hash = generate_search_hash(cache_params)
    print(f"Search hash: {search_hash} (chain={'+'.join(chain)})")

    def transform_for(name: str):
        if name == "bytemine":
            return transform_bytemine_profile
        if name == "crustdata":
            return transform_crustdata_profile
        if degraded:
            return lambda profile, _params=None: transform_preview_profile(profile)
        return transform_wiza_contact

    transform = transform_for(prov)

    campaign_keys: set[str] = set()
    campaign_exclusions: list[str] = []
    if prov == "crustdata" and request.campaign_id:
        campaign_keys, campaign_exclusions = await campaign_seen(request.campaign_id)

    exclusions = list(dict.fromkeys(
        (request.exclude_profiles or []) + campaign_exclusions
    ))

    # Campaign membership changes after every response, so campaign searches
    # must never reuse a shared cached page. Explicit refresh also bypasses it.
    use_cache = not request.refresh and not request.campaign_id

    # Cache is best-effort: a DB outage must not blank out every search.
    cached = await cache_lookup(search_hash) if use_cache else None
    if cached:
        print(f"Cache HIT for hash: {search_hash}")
        cached_payload = json.loads(cached.results)
        served_by = prov
        if isinstance(cached_payload, dict):
            data = cached_payload.get("profiles") or []
            total = cached_payload.get("total", len(data))
            next_cursor = cached_payload.get("next_cursor")
            # Rows written before the chain existed have no provider recorded;
            # they can only have come from the head of the chain.
            served_by = cached_payload.get("provider") or prov
            transform = transform_for(served_by)
        else:
            # Backwards compatibility with cache rows written before cursor
            # metadata was persisted.
            data = cached_payload
            total = len(data)
            next_cursor = None
        leads = [transform(r, params) for r in data]
        return SearchResponse(
            success=True, source="cache", from_cache=True,
            count=len(leads), total=total, leads=leads, data=data,
            next_cursor=next_cursor, campaign_id=request.campaign_id,
            provider=served_by,
        )

    print(f"Cache MISS — walking chain {'+'.join(chain)}")

    async def run_provider(name: str):
        """One provider's search. Returns (raw_results, total, next_cursor)."""
        if name == "bytemine":
            result = await bytemine_person_search(
                params, max(min(params.get("limit", 10), 100), 1),
                cursor=request.cursor, offset=request.start_offset)
            return result["profiles"], result["total"], result.get("next_cursor")

        if name == "crustdata":
            # The title-only guard belongs to Crustdata's filter model, so it is
            # applied when Crustdata runs rather than up front — a chain that
            # never reaches Crustdata should not be refused by its rules.
            enforce_crustdata_search_scope(params, request.allow_broad_search)
            page_size = max(min(params.get("limit", 10), 100), 1)
            # Crustdata pages by cursor, not offset. When the caller asks for a
            # numbered page without one, over-fetch and slice rather than hand
            # back the first page again. Its ceiling is 100 rows per call, so
            # past that the only honest answer is to ask for the cursor.
            skip = 0 if request.cursor else request.start_offset
            if skip:
                if skip + page_size > 100:
                    raise HTTPException(
                        status_code=422,
                        detail=("Crustdata pages beyond this point need the "
                                "`next_cursor` from the previous response rather "
                                "than a page number."),
                    )
                fetch_size = skip + page_size
            else:
                fetch_size = page_size
            result = await crustdata_person_search(
                params,
                fetch_size,
                cursor=request.cursor,
                exclude_profiles=exclusions,
            )
            found = dedupe_crustdata_profiles(result["profiles"])
            if skip:
                found = found[skip:]
            # Provider exclusions require profile URLs; this catches previously
            # seen stable IDs even when a profile has no URL.
            if campaign_keys:
                found = [p for p in found if _profile_lead_key(p) not in campaign_keys]
            if request.campaign_id:
                found = await record_new_campaign_profiles(request.campaign_id, found)
            return found, result["total"], result.get("next_cursor")

        if degraded:
            # Credit-free preview search — see provider_state(). Wiza caps this
            # endpoint at 30 profiles per call.
            skip = request.start_offset
            data = await wiza_prospect_search(
                params, max(min(params.get("limit", 10) + skip, 30), 1))
            found = data.get("profiles") or []
            return (found[skip:] if skip else found), data.get("total", len(found)), None

        skip = request.start_offset
        found = await fetch_from_wiza(
            {**params, "limit": params.get("limit", 10) + skip} if skip else params)
        total = len(found)
        return (found[skip:] if skip else found), total, None

    raw_results: list = []
    total = 0
    next_cursor = None
    served_by = chain[0]
    attempts: list = []
    ran = False
    refused: dict[str, str] = {}

    for index, name in enumerate(chain):
        is_last = index == len(chain) - 1
        try:
            raw_results, total, next_cursor = await run_provider(name)
            ran = True
        except ProviderUnsupported as unsupported:
            # The provider cannot express one of the requested filters. Moving on
            # keeps the ICP intact; dropping the filter would not.
            refused[name] = unsupported.field
            print(f"{name} cannot express {unsupported} — trying next provider")
            attempts.append({"provider": name, "outcome": "unsupported_filter",
                             "detail": unsupported.field})
            continue
        except HTTPException as exc:
            # 4xx about the request itself will fail identically everywhere, so
            # they surface rather than burning a call on every provider.
            if exc.status_code in (400, 422) or is_last:
                raise
            print(f"{name} failed ({exc.status_code}) — trying next provider")
            attempts.append({"provider": name, "outcome": "error",
                             "detail": exc.status_code})
            continue

        if raw_results:
            served_by = name
            break

        # An empty result is not an error, but the next provider may hold the
        # data this one lacks. Cheap to try: providers charge per result.
        attempts.append({"provider": name, "outcome": "no_results"})
        if not is_last:
            print(f"{name} returned 0 leads — trying next provider")
        else:
            served_by = name

    # Every provider refused, so nothing actually ran. Returning an empty
    # success here would read as "no such leads exist" when the truth is that no
    # configured provider can express the filter — a different answer, and the
    # only one that tells the user what to change.
    if not ran:
        fields = sorted({field for field in refused.values()})
        raise HTTPException(
            status_code=422,
            detail=(
                f"No configured provider can search by {', '.join(fields)}. "
                f"Tried {', '.join(f'{p} ({f})' for p, f in refused.items())}. "
                "Your search credit was not used — remove that filter or add a "
                "provider that supports it."
            ),
        )

    transform = transform_for(served_by)
    print(f"{served_by}{' (degraded preview)' if degraded and served_by == 'wiza' else ''} "
          f"returned {len(raw_results)} leads")

    leads = [transform(r, params) for r in raw_results]
    cache_payload = {"profiles": raw_results, "total": total,
                     "next_cursor": next_cursor, "provider": served_by}
    if not request.campaign_id:
        await cache_store(search_hash, cache_params, cache_payload)

    return SearchResponse(
        success=True, source="api", from_cache=False,
        count=len(leads), total=total, leads=leads, data=raw_results,
        next_cursor=next_cursor, campaign_id=request.campaign_id,
        provider=served_by, provider_attempts=attempts or None,
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
    prov = active_provider()
    size = max(min(request.limit or 10, 30), 0)

    async def fetcher():
        if prov == "crustdata":
            data = await crustdata_person_search(params, size)
        else:
            try:
                data = await wiza_prospect_search(params, size)
            except ProviderUnsupported as unsupported:
                # There is no chain to fall through to here, so say plainly that
                # this provider cannot size that audience rather than returning
                # a count for a broader search than the one that was asked for.
                raise HTTPException(
                    status_code=422,
                    detail=(f"{prov} cannot preview a search filtered by "
                            f"{unsupported.field}. Run the full search instead, "
                            "which can use a provider that supports it."),
                ) from unsupported
        # Store total alongside profiles so it survives caching
        return [{"_total": data.get("total", 0)}] + (data.get("profiles") or [])

    raw, from_cache = await cached_or_fetch(
        "preview", {**params, "_provider": prov, "_size": size}, fetcher)

    total = raw[0].get("_total", 0) if raw and "_total" in raw[0] else 0
    profiles = [r for r in raw if "_total" not in r]
    preview_transform = transform_crustdata_profile if prov == "crustdata" else transform_preview_profile
    leads = [preview_transform(p) for p in profiles]

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
    prov = active_provider()
    # Pull the max preview window so we can dedupe as many companies as possible
    size = 30
    enrich_limit = max(min(enrich_limit, 30), 0)

    async def fetcher():
        if prov == "crustdata":
            # Crustdata returns firmographics inline, so companies are fully
            # populated from the people search — no per-company enrich needed.
            data = await crustdata_person_search(params, size)
            return aggregate_companies_crustdata(data.get("profiles") or [])

        try:
            data = await wiza_prospect_search(params, size)
        except ProviderUnsupported as unsupported:
            # No chain here either — a company list built without the filter
            # would be a list of the wrong companies.
            raise HTTPException(
                status_code=422,
                detail=(f"{prov} cannot search companies by {unsupported.field}. "
                        "Remove that filter or configure a provider that supports it."),
            ) from unsupported
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

    cache_params = {**params, "_provider": prov, "_enrich": enrich, "_enrich_limit": enrich_limit}
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

def _cached_items(payload) -> list:
    """Read both legacy list rows and cursor-aware search cache payloads."""
    if isinstance(payload, dict) and isinstance(payload.get("profiles"), list):
        return payload["profiles"]
    return payload if isinstance(payload, list) else []


def _is_crustdata_profile(value: dict) -> bool:
    return bool(value.get("crustdata_person_id") is not None or value.get("basic_profile"))


@app.delete("/campaigns/{campaign_id}/seen")
async def clear_campaign_seen(campaign_id: str):
    """Reset a campaign's dedupe ledger so its leads may be surfaced again."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", campaign_id):
        raise HTTPException(status_code=422, detail="Invalid campaign_id")
    async with async_session() as session:
        result = await session.execute(
            CampaignSeenLead.__table__.delete().where(CampaignSeenLead.campaign_id == campaign_id)
        )
        await session.commit()
    return {"campaign_id": campaign_id, "deleted": result.rowcount or 0}

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
        deleted = sum(1 for c in all_cached if len(_cached_items(json.loads(c.results))) == 0)
        for c in all_cached:
            if len(_cached_items(json.loads(c.results))) == 0:
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
            "ttl_seconds": settings.search_cache_ttl_seconds,
            "recent_searches": [
                {
                    "search_hash": s.search_hash,
                    "params": json.loads(s.search_params),
                    "result_count": len(_cached_items(json.loads(s.results))),
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
            data = _cached_items(json.loads(cached.results))
            for contact in data:
                key = (_profile_lead_key(contact) if _is_crustdata_profile(contact)
                       else f"{contact.get('full_name', '')}-{contact.get('name', '')}")
                if key not in seen:
                    seen.add(key)
                    transform = (transform_crustdata_profile if _is_crustdata_profile(contact)
                                 else transform_wiza_contact)
                    all_leads.append(transform(contact, search_params))

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

        payload = json.loads(cached.results)
        data = _cached_items(payload)
        search_params = json.loads(cached.search_params)
        transform = (transform_crustdata_profile
                     if data and _is_crustdata_profile(data[0]) else transform_wiza_contact)
        leads = [transform(r, search_params) for r in data]
        return {
            "success": True, "from_cache": True,
            "search_params": search_params,
            "leads": leads, "data": data,
            "count": len(leads), "total": len(leads),
            "next_cursor": payload.get("next_cursor") if isinstance(payload, dict) else None,
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
                    "result_count": len(_cached_items(json.loads(c.results))),
                    "sample_lead": (_cached_items(json.loads(c.results))[0]
                                    if _cached_items(json.loads(c.results)) else None),
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in all_cached
            ],
        }


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
