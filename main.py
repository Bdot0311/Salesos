"""
Cache-First Lead Generation Proxy
A FastAPI application that caches Wiza API results to reduce costs.
"""

import asyncio
import base64
import contextvars
import hashlib
import hmac
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import anthropic
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field, AliasChoices
from pydantic_settings import BaseSettings
from sqlalchemy.exc import IntegrityError
from sqlalchemy import BigInteger, Column, String, Text, DateTime, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    database_url: str
    wiza_api_key: Optional[str] = None
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
    coldiq_api_key: Optional[str] = None
    getleads_api_key: Optional[str] = None
    crustdata_api_key: Optional[str] = None
    # treg is an all-in-one tool catalog. We use only its routed lead-gen
    # endpoints, while treg itself chooses among the eligible upstream tools.
    # Keep this org-scoped token on the backend: it pays for every customer.
    findymail_api_key: Optional[str] = None
    fiber_api_key: Optional[str] = None
    treg_token: Optional[str] = None
    treg_org_id: Optional[str] = None
    treg_base_url: str = "https://treg.to"
    # Used only when the caller has neither X-Customer-ID nor an authenticated
    # identity. Prefer setting this to the product/account slug in server-to-
    # server deployments. Browser traffic otherwise receives a pseudonymous,
    # stable client identifier so Treg calls are never left unattributed.
    treg_default_customer_id: Optional[str] = None
    # Protects the invoice and budget management endpoints below. It is
    # intentionally separate from TREG_TOKEN so callers never receive treg's
    # credential or team-level balance details.
    billing_admin_key: Optional[str] = None
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

    # Credentials arrive from a dashboard's paste box, and a stray space or
    # newline on the end survives the round trip. httpx then refuses to build the
    # header at all — "Illegal header value b'Bearer ciq_live_… '" — so every
    # call to that provider fails, while the key still reads as configured and
    # the provider stays in the chain looking healthy.
    #
    # That is exactly what happened to ColdIQ: one trailing space meant it never
    # returned a single lead, and the email verification that runs through it
    # answered "unknown" for every address, for as long as it had been deployed.
    #
    # This applies to every field rather than a named list. The named list said
    # it "cannot be forgotten by the next one" and was then forgotten by the very
    # next one: GetLeads was added with the same trailing space, failed on every
    # call for the same reason, and the validator did not cover it because nobody
    # went back to add it. A list of fields to protect is a list someone has to
    # remember to extend; "*" is not.
    @field_validator("*", mode="after")
    @classmethod
    def _strip_credential(cls, value):
        return value.strip() if isinstance(value, str) else value

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
TREG_META_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

# Default order, best-fit first. Filtered down to configured providers by
# provider_chain().
# getleads sits ahead of coldiq: it filters on industry, headcount and region
# for real, where coldiq can only rank on them. See build_getleads_filters.
# findymail sits late on purpose. Its search is asynchronous — submit, poll,
# then fetch — so it is the slowest leg by a wide margin, and it bills per
# result found. In a sequential waterfall that makes it the right answer only
# once the fast, cheaper providers have come back empty.
# fiber sits right after getleads: it is the only leg that can express this
# product's whole ICP — title, industry, country and headcount — against
# LinkedIn's own vocabulary, and it pages by cursor rather than offset. It is
# billed per profile returned, like getleads, so it goes after the cheaper
# masked-search providers and ahead of the per-record reveal tools.
PROVIDER_ORDER = ("bytemine", "crustdata", "getleads", "fiber", "treg",
                  "coldiq", "findymail", "wiza")


def provider_configured(name: str) -> bool:
    """True when this provider has the credentials to be called at all."""
    if name == "bytemine":
        return bool(settings.bytemine_api_key)
    if name == "crustdata":
        return bool(settings.crustdata_api_key)
    if name == "coldiq":
        return bool(settings.coldiq_api_key)
    if name == "getleads":
        return bool(settings.getleads_api_key)
    if name == "treg":
        return bool(settings.treg_token)
    if name == "findymail":
        return bool(settings.findymail_api_key)
    if name == "fiber":
        return bool(settings.fiber_api_key)
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
    return chain


def linkedin_identity(url) -> Optional[str]:
    """The part of a LinkedIn URL that identifies the person.

    One definition, because "have we shown this person already?" is asked in
    several places and every spelling of the same profile has to answer the
    same: http vs https, with or without www, with or without a trailing slash.
    ColdIQ's search returns https and its enrich returns http for the very same
    person, so a raw string comparison misses.
    """
    text = str(url or "").strip().lower()
    if not text or "linkedin.com/" not in text:
        return None
    return text.rstrip("/").split("linkedin.com/")[-1] or None


def _dedupe_key(lead: dict) -> str:
    """The strongest identity a transformed lead offers.

    Providers overlap — the same person can come back from two of them with
    different field names and different completeness — so merged results are
    deduped on what actually identifies a person, in order of how much it
    proves: a LinkedIn URL, then a work email, then name plus company.
    """
    linkedin = linkedin_identity(lead.get("linkedin_url"))
    if linkedin:
        return "li:" + linkedin
    email = str(lead.get("contact_email") or lead.get("business_email") or "").strip().lower()
    if email:
        return "em:" + email
    name = str(lead.get("contact_name") or "").strip().lower()
    company = str(lead.get("company_name") or "").strip().lower()
    return f"nc:{name}|{company}"


def merge_provider_results(buckets: list, transform_for) -> tuple[list, list, int, dict]:
    """Fold every provider's results into one list.

    Returns (raw_rows, leads, total, cursors). Order is round-robin across the
    providers that returned something rather than one provider's whole page
    followed by another's: a merged page should not be all Bytemine at the top
    just because it is first in the chain.

    A duplicate keeps the copy that arrived first under that ordering, and the
    total is summed across providers less the duplicates removed, so it stays
    consistent with what was actually returned.
    """
    seen: set = set()
    raw_rows: list = []
    leads: list = []
    cursors: dict = {}
    duplicates = 0

    for bucket in buckets:
        if bucket.get("next_cursor"):
            cursors[bucket["provider"]] = bucket["next_cursor"]

    live = [b for b in buckets if b.get("profiles")]
    depth = max((len(b["profiles"]) for b in live), default=0)
    for index in range(depth):
        for bucket in live:
            profiles = bucket["profiles"]
            if index >= len(profiles):
                continue
            raw = profiles[index]
            lead = transform_for(bucket["provider"])(raw)
            key = _dedupe_key(lead)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            raw_rows.append(raw)
            leads.append(lead)

    total = max(sum(int(b.get("total") or 0) for b in buckets) - duplicates, len(leads))
    return raw_rows, leads, total, cursors


def encode_cursors(cursors: dict) -> Optional[str]:
    """Pack per-provider cursors into the one opaque string the API exposes.

    Every provider pages independently, so a merged search has one cursor per
    contributor rather than one overall. Callers keep treating `next_cursor` as
    an opaque token they echo back; only this function and its decoder know it
    carries several.
    """
    live = {name: cur for name, cur in (cursors or {}).items() if cur}
    if not live:
        return None
    raw = json.dumps(live, sort_keys=True).encode()
    return "multi:" + base64.urlsafe_b64encode(raw).decode()


def decode_cursors(cursor: Optional[str], chain: list) -> dict:
    """Unpack a cursor into {provider: cursor}.

    A plain string predates the merged search and can only have come from the
    provider that led the chain, so it is handed to that one alone — giving it
    to every provider would ask each to resume from a position in someone
    else's result set.
    """
    if not cursor:
        return {}
    if cursor.startswith("multi:"):
        try:
            decoded = json.loads(base64.urlsafe_b64decode(cursor[6:]).decode())
            if isinstance(decoded, dict):
                return {k: v for k, v in decoded.items() if isinstance(v, str) and v}
        except Exception as exc:
            print(f"WARNING: unreadable cursor ({exc}) — starting from the first page")
            return {}
    return {chain[0]: cursor} if chain else {}


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
    head = chain[0] if chain else "none"
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
        # The trailing guard also rejects a term that is the stem of a filename.
        # Every pitch in production mentions "llm.txt robot.txt" as an SEO
        # deliverable, and "llm" was matching inside it — so a search for salon
        # owners went out to GetLeads as company_description "llm" and to Fiber
        # as a keyword, narrowing every one of those searches to companies that
        # talk about language models.
        if not re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9]|\.[a-z])", low):
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
    # semantic_query is deliberately NOT refused. It is the sentence the
    # structured filters were parsed out of, not an additional criterion the
    # user stated — those filters are already in this body. Refusing it made
    # Wiza sit out every search from the UI, which always sends a query, so a
    # three-provider chain quietly became a one-provider chain. Wiza cannot rank
    # by the sentence; it searches the filters, which is a real contribution.

    # Seniority — only add if we have a known Wiza level, and only when the job
    # title does not already imply it (see seniority_implied_by_title). Wiza's
    # taxonomy happens to agree with ours on Founder/Owner where Bytemine's and
    # Crustdata's do not, so this leg was not the one losing people — but the
    # filter is redundant here too, and a redundant AND against someone else's
    # taxonomy is exactly the shape of that bug.
    if p.get("seniority") and not seniority_implied_by_title(
            p.get("job_title"), p["seniority"]):
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

    # Wiza buckets a location as country, state or city. A region is none of
    # those, and location_filter would fall through to the city bucket — the
    # same silent no-match that sent cities:["Europe"] to Bytemine. Refused so
    # the fan-out reaches Crustdata, which does match a region by name.
    for key in ("location", "company_location"):
        if p.get(key) and classify_location(str(p[key]))[0] == "region":
            raise ProviderUnsupported(key, p[key])

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


class TregUsage(Base):
    """Local audit copy of Treg's authoritative append-only billing ledger.

    `treg_call_id` makes retries idempotent on our side too. Invoices still use
    treg's usage/by-tag ledger; these rows join a product request to that ledger
    and make support/debugging possible without relying on Treg's call log.
    """

    __tablename__ = "treg_usage"

    treg_call_id = Column(String(128), primary_key=True)
    customer_id = Column(String(128), nullable=False, index=True)
    workspace_id = Column(String(128), nullable=True, index=True)
    feature = Column(String(64), nullable=False)
    endpoint_id = Column(String(128), nullable=False)
    cost_micro = Column(BigInteger, nullable=False, default=0)
    served_by = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# =============================================================================
# Database Setup
# =============================================================================

engine = create_async_engine(settings.async_database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =============================================================================
# Treg customer attribution and billing
# =============================================================================

_treg_request_context = contextvars.ContextVar("treg_request_context", default=None)


def _valid_treg_meta_value(name: str, value: Optional[str], required: bool = False) -> Optional[str]:
    value = value.strip() if isinstance(value, str) else value
    if not value:
        if required:
            raise HTTPException(status_code=400, detail=f"Missing X-{name.replace('_', '-').title()}")
        return None
    if not TREG_META_VALUE_RE.fullmatch(value) or "@" in value:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be 1-128 URL-safe characters and cannot be an email",
        )
    return value


def treg_request_context() -> dict:
    """Return validated server-owned billing context for the current request."""
    ctx = _treg_request_context.get() or {}
    return {
        "customer_id": _valid_treg_meta_value(
            "customer_id", ctx.get("customer_id"), required=True),
        "workspace_id": _valid_treg_meta_value(
            "workspace_id", ctx.get("workspace_id")),
        "idempotency_key": ctx.get("idempotency_key"),
    }


def _jwt_subject(authorization: Optional[str]) -> Optional[str]:
    """Read a stable subject from an already-authenticated bearer JWT.

    This does not authenticate the request; that remains the API gateway's job.
    It only avoids making a rotating access token itself the billing identity.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(None, 1)[1].strip()
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded).decode())
    except Exception:
        return None
    for key in ("tenant_id", "organization_id", "org_id", "sub", "user_id"):
        value = claims.get(key) if isinstance(claims, dict) else None
        if value:
            return str(value)
    return None


def treg_customer_id_from_request(request: Request) -> str:
    """Resolve the most specific stable billing identity available.

    Explicit tenant headers win. Authenticated JWT subjects are next. The final
    fallback is a one-way client fingerprint (never the raw IP or user agent),
    which keeps production searches working and attributable even for the
    current caller that sends no tenant header.
    """
    explicit = request.headers.get("X-Customer-ID")
    if explicit:
        return explicit

    subject = _jwt_subject(request.headers.get("Authorization"))
    if subject:
        digest = hashlib.sha256(subject.encode()).hexdigest()[:24]
        return f"auth_{digest}"

    if settings.treg_default_customer_id:
        return settings.treg_default_customer_id

    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",", 1)[0].strip()
    peer = forwarded or (request.client.host if request.client else "unknown")
    agent = request.headers.get("User-Agent") or "unknown"
    digest = hashlib.sha256(f"{peer}|{agent}".encode()).hexdigest()[:24]
    return f"anon_{digest}"


def _treg_meta_header(ctx: dict, feature: str) -> str:
    tags = [f"customer={ctx['customer_id']}", f"feature={feature}"]
    if ctx.get("workspace_id"):
        tags.insert(1, f"workspace={ctx['workspace_id']}")
    return ", ".join(tags)


async def record_treg_usage(
    *, call_id: Optional[str], ctx: dict, feature: str, endpoint_id: str,
    cost_micro: int, served_by: Optional[str],
) -> None:
    if not call_id:
        return
    try:
        async with async_session() as session:
            values = dict(
                treg_call_id=call_id,
                customer_id=ctx["customer_id"],
                workspace_id=ctx.get("workspace_id"),
                feature=feature,
                endpoint_id=endpoint_id,
                cost_micro=cost_micro,
                served_by=served_by,
            )
            statement = pg_insert(TregUsage).values(**values).on_conflict_do_nothing(
                index_elements=[TregUsage.treg_call_id])
            await session.execute(statement)
            await session.commit()
    except Exception as exc:
        # Treg's ledger remains the billing source of truth. A local audit write
        # must not turn a paid, successful provider response into a user error.
        print(f"WARNING: could not store treg usage row {call_id}: {exc}")


def _safe_treg_error(response: httpx.Response) -> HTTPException:
    """Redact team balance/top-up details while preserving customer cap errors."""
    try:
        body = response.json()
    except Exception:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    is_treg_refusal = response.headers.get("X-Treg-Error") == "1"
    if error in ("tag_blocked", "tag_spend_cap_reached"):
        return HTTPException(status_code=response.status_code, detail=body)
    if response.status_code == 422:
        return HTTPException(status_code=422, detail=body.get("detail", "Invalid lead request"))
    if response.status_code == 429:
        return HTTPException(status_code=503, detail="Lead data provider is temporarily rate limited")
    if response.status_code == 402:
        return HTTPException(status_code=503, detail="Lead data provider billing is temporarily unavailable")
    if not is_treg_refusal:
        return HTTPException(
            status_code=response.status_code if response.status_code < 500 else 502,
            detail=body.get("detail", "Upstream lead provider request failed"),
        )
    return HTTPException(
        status_code=response.status_code if response.status_code < 500 else 502,
        detail=body.get("detail", "Lead data provider request failed"),
    )


async def treg_call(
    endpoint_id: str, payload: dict, feature: str, query: Optional[dict] = None,
) -> dict:
    """The only product call path to Treg: tagged, metered and auditable."""
    if not settings.treg_token:
        raise HTTPException(status_code=503, detail="Treg is not configured")
    ctx = treg_request_context()
    headers = {
        "X-Treg-Token": settings.treg_token,
        "X-Treg-Meta": _treg_meta_header(ctx, feature),
        "Content-Type": "application/json",
    }
    if ctx.get("idempotency_key"):
        headers["Idempotency-Key"] = ctx["idempotency_key"]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.treg_base_url.rstrip('/')}/call/{endpoint_id}",
            headers=headers,
            params=query,
            json=payload,
        )

    call_id = response.headers.get("X-Treg-Call-Id")
    try:
        cost_micro = int(response.headers.get("X-Treg-Cost-Micro") or 0)
    except ValueError:
        cost_micro = 0
    await record_treg_usage(
        call_id=call_id,
        ctx=ctx,
        feature=feature,
        endpoint_id=endpoint_id,
        cost_micro=cost_micro,
        served_by=response.headers.get("X-Treg-Served-By"),
    )

    if response.is_error:
        raise _safe_treg_error(response)
    try:
        return response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Treg returned an invalid response") from exc


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
# Phone numbers
# =============================================================================
#
# Every provider returns a number under a different key, and each of those keys
# already says what kind of line it is — Wiza splits mobile_phone from
# phone_number, Bytemine splits mobile_phone from direct_dial. We were reading
# whichever key happened to be first and throwing that distinction away, so a
# switchboard and a personal mobile arrived indistinguishable.
#
# The distinction is worth keeping for two reasons. A mobile is the more useful
# number to a salesperson. And WhatsApp runs on mobile numbers only, so "is this
# a mobile" is the closest thing to a WhatsApp signal these providers actually
# sell: none of them expose a WhatsApp flag, and checking numbers against
# WhatsApp in bulk is not something its API permits. "mobile" is an honest
# answer to that question; "has WhatsApp" would not be.

_MOBILE_KEYS = ("mobile_phone", "mobile", "cell_phone", "cell", "personal_phone")
_OFFICE_KEYS = ("direct_dial", "phone_number", "work_phone", "office_phone",
                "company_phone", "hq_phone")
# Keys that carry a number without saying which kind it is.
_UNTYPED_KEYS = ("phone", "telephone", "contact_phone")

_PHONE_TYPE_WORDS = (
    ("mobile", "mobile"), ("cell", "mobile"), ("personal", "mobile"),
    ("direct", "office"), ("landline", "office"), ("office", "office"),
    ("work", "office"), ("company", "office"), ("business", "office"),
)


def classify_phone_type(value) -> Optional[str]:
    """Map a provider's line-type word onto "mobile" or "office".

    None when the word is missing or unrecognised — an unknown line type is
    reported as unknown rather than guessed at, because the whole value of the
    field is that "mobile" can be trusted.
    """
    if not value:
        return None
    word = str(value).strip().lower()
    for token, kind in _PHONE_TYPE_WORDS:
        if token in word:
            return kind
    return None


def clean_phone(value) -> Optional[str]:
    """Return a real number, or None for a missing or masked one.

    Bytemine returns withheld numbers as asterisks from search; an unlock that
    fails quietly can leave those in place, and a string of asterisks stored as
    a phone number is worse than an empty field.
    """
    text = str(value).strip() if value is not None else ""
    if not text or "*" in text:
        return None
    return text


def pick_phone(contact: dict) -> tuple:
    """Choose the best number on a provider record and say what kind it is.

    Returns (number, type) with type "mobile", "office", or None when a number
    came with nothing to identify it. A mobile wins over an office line when a
    contact has both.
    """
    c = contact or {}
    mobile = office = untyped = None

    # An array of numbers carries its own per-entry type. Wiza's list search
    # calls it `phones`, Crustdata's enrich calls it `phone_numbers`.
    entries = list(c.get("phones") or []) + list(c.get("phone_numbers") or [])
    for entry in entries:
        if not isinstance(entry, dict):
            number = clean_phone(entry)
            if number and not untyped:
                untyped = number
            continue
        number = clean_phone(entry.get("pretty_number") or entry.get("number"))
        if not number:
            continue
        kind = classify_phone_type(
            entry.get("type") or entry.get("phone_type") or entry.get("line_type"))
        if kind == "mobile" and not mobile:
            mobile = number
        elif kind == "office" and not office:
            office = number
        elif not kind and not untyped:
            untyped = number

    # Flat keys, where the key name is itself the line type.
    for key in _MOBILE_KEYS:
        if not mobile:
            mobile = clean_phone(c.get(key))
    for key in _OFFICE_KEYS:
        if not office:
            office = clean_phone(c.get(key))
    for key in _UNTYPED_KEYS:
        if not untyped:
            untyped = clean_phone(c.get(key))

    if mobile:
        return mobile, "mobile"
    if office:
        return office, "office"
    return untyped, None


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

    primary_phone, phone_type = pick_phone(contact)

    return {
        "contact_name": contact.get("full_name"),
        "first_name": contact.get("first_name"),
        "last_name": contact.get("last_name"),
        "job_title": contact.get("title"),
        "linkedin_url": contact.get("linkedin"),
        "business_email": primary_email,
        "email_status": contact.get("email_status"),
        "phone": primary_phone,
        "phone_type": phone_type,
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

    primary_phone, phone_type = pick_phone(contact)

    return {
        "contact_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "job_title": contact.get("title"),
        "linkedin_url": contact.get("linkedin_profile_url") or contact.get("linkedin"),
        "business_email": primary_email,
        "email_status": contact.get("email_status"),
        "phone": primary_phone,
        "phone_type": phone_type,
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

def wiza_no_contacts(response: httpx.Response) -> bool:
    """Wiza's empty-export sentinel is a successful zero-result outcome."""
    return (response.status_code == 400
            and "no contacts to export" in response.text.lower())

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

        # Wiza reports an empty completed list as HTTP 400. That is a valid
        # zero-result search, not a broken provider and not a reason to retry a
        # second export segment that will return the same response.
        if wiza_no_contacts(contacts_resp):
            print("Wiza returned 0 contacts")
            return []

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

# Named separately from the parser's _US_STATES (full lowercase names, line ~262)
# which this used to shadow at module scope.
_US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

_US_STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}


# Country name (or code) -> ISO 3166 alpha-2. /contacts/search has no country
# field, so these are refused there rather than searched as a city called
# "Germany"; /b2b-search does have one and uses the code.
_BM_COUNTRY_CODE = {
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states": "US", "america": "US",
    "gb": "GB", "uk": "GB", "united kingdom": "GB", "great britain": "GB",
    "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "ca": "CA", "canada": "CA", "au": "AU", "australia": "AU",
    "de": "DE", "germany": "DE", "fr": "FR", "france": "FR",
    "es": "ES", "spain": "ES", "it": "IT", "italy": "IT",
    "nl": "NL", "netherlands": "NL", "se": "SE", "sweden": "SE",
    "no": "NO", "norway": "NO", "dk": "DK", "denmark": "DK",
    "fi": "FI", "finland": "FI", "ch": "CH", "switzerland": "CH",
    "at": "AT", "austria": "AT", "be": "BE", "belgium": "BE",
    "ie": "IE", "ireland": "IE", "in": "IN", "india": "IN",
    "sg": "SG", "singapore": "SG", "il": "IL", "israel": "IL",
    "br": "BR", "brazil": "BR", "mx": "MX", "mexico": "MX",
    "ar": "AR", "argentina": "AR", "co": "CO", "colombia": "CO",
    "cl": "CL", "chile": "CL", "jp": "JP", "japan": "JP",
    "kr": "KR", "south korea": "KR", "korea": "KR",
    "cn": "CN", "china": "CN", "tw": "TW", "taiwan": "TW",
    "nz": "NZ", "new zealand": "NZ", "za": "ZA", "south africa": "ZA",
    "ng": "NG", "nigeria": "NG", "ke": "KE", "kenya": "KE",
    "pl": "PL", "poland": "PL", "pt": "PT", "portugal": "PT",
    "cz": "CZ", "czechia": "CZ", "czech republic": "CZ",
    "ro": "RO", "romania": "RO", "hu": "HU", "hungary": "HU",
    "ua": "UA", "ukraine": "UA", "ae": "AE", "united arab emirates": "AE",
    "sa": "SA", "saudi arabia": "SA", "tr": "TR", "turkey": "TR",
    "ru": "RU", "russia": "RU",
}

# Two-letter codes that name a US state *and* a country we recognise — CA, DE,
# IN, IL and so on. They are read as countries, because that is what produces
# them: fetch-external-leads sets `location` from normalizeCountry(), so a
# two-letter location is always an ISO country code. Reading "CA" as California
# meant every search for Canada came back from the wrong continent. A US state
# arrives by name ("Texas"), which is unambiguous.
#
# Derived rather than listed: written by hand it drifts from the country map it
# describes, and a code in one but not the other is a silent misread.
_STATE_CODE_IS_ALSO_A_COUNTRY = _US_STATE_CODES & set(_BM_COUNTRY_CODE.values())

# Reverse lookups, for providers whose filters take names where classify_location
# hands back a code. Derived from the maps above rather than written out again,
# so a country added in one place cannot go missing here.
#
# The longest spelling of each code wins, because the aliases are abbreviations
# of the real name: "us", "usa" and "united states" all map to US, and only the
# last of those is what a country-name filter will match.
_GL_COUNTRY_NAME: dict = {}
for _name, _code in sorted(_BM_COUNTRY_CODE.items(), key=lambda kv: -len(kv[0])):
    _GL_COUNTRY_NAME.setdefault(_code, _name.title())
_US_STATE_CODE_TO_NAME = {
    _code: _name.title() for _name, _code in _US_STATE_NAME_TO_CODE.items()
}


# Continents and multi-country regions. An ICP routinely names one — "founders
# in Europe", "APAC", "the Nordics" — and none of these providers has a field
# for it: they filter by country, state or city.
#
# Without this set they fell through to "city", so "Europe" was sent as
# cities:["Europe"] to Bytemine and city:"Europe" to /b2b-search. No company is
# in a city called Europe, so those searches returned total_companies 0 while
# reporting success. A region has to be recognised as a region precisely so it
# can be refused: the fan-out then reaches Crustdata, whose location field is a
# text match over the full location string and does match "Europe".
_REGIONS = frozenset({
    "africa", "americas", "antarctica", "asia", "asia pacific", "asia-pacific",
    "apac", "anz", "australasia", "benelux", "british isles", "caribbean",
    "central america", "central asia", "cis", "dach", "eastern europe", "emea",
    "europe", "eu", "european union", "iberia", "latam", "latin america",
    "mena", "middle east", "middle east and africa", "nordics", "nordic",
    "north america", "northern europe", "oceania", "scandinavia",
    "south america", "south asia", "southeast asia", "south east asia",
    "sub-saharan africa", "western europe", "worldwide", "global",
})


def classify_location(token: str) -> tuple[str, str]:
    """Read one location string as a region, US state, country or city.

    Returns (kind, value) where kind is "region" (a continent or multi-country
    bloc, no provider has a field for it), "state" (2-letter code), "country"
    (ISO alpha-2), "city" (the token as given), or "unknown" for a two-letter
    token that is neither a state nor a country we recognise.

    Regions are checked first: "Europe" is not a city, and treating it as one
    is a filter that matches nothing while looking like it worked.

    State names are checked before country codes so "Georgia" the state and "GA"
    the code do not collide, and two-letter codes resolve to countries — see
    _STATE_CODE_IS_ALSO_A_COUNTRY for why.
    """
    token = (token or "").strip()
    lowered = token.lower()
    upper = token.upper()

    if lowered in _REGIONS:
        return "region", token
    state_code = _US_STATE_NAME_TO_CODE.get(lowered)
    if state_code:
        return "state", state_code
    if lowered in _BM_COUNTRY_CODE:
        return "country", _BM_COUNTRY_CODE[lowered]
    if upper in _US_STATE_CODES:
        return "state", upper
    if len(token) > 2:
        return "city", token
    return "unknown", token


# One seniority idea, several spellings. _SENIORITY_RULES emits canonical keys
# while callers hand us whatever the parser or the caller wrote, so both sides
# are folded onto the same word before being compared. Without this, "CEO" +
# seniority "cxo" reads as two different criteria when it is one.
#
# founder and owner are deliberately the same bucket: every provider here maps
# them to a single level (Bytemine "Owner", Crustdata "Owner", ColdIQ "owner"),
# so a title of "Founder" does imply a stated seniority of "owner".
_CANONICAL_SENIORITY = {
    "founder": "owner", "co-founder": "owner", "cofounder": "owner",
    "owner": "owner", "partner": "partner",
    "cxo": "c_suite", "c-level": "c_suite", "c-suite": "c_suite",
    "c_suite": "c_suite", "executive": "c_suite", "chief": "c_suite",
    "vp": "vp", "vice president": "vp", "svp": "vp", "evp": "vp",
    "director": "director", "head": "manager", "manager": "manager",
    "senior": "senior", "sr": "senior",
    "entry": "junior", "junior": "junior", "intern": "junior",
    "training": "junior",
}


def _canonical_seniority(value: str) -> str:
    key = (value or "").strip().lower()
    return _CANONICAL_SENIORITY.get(key, key)


def seniority_implied_by_title(job_title: str, seniority: str) -> bool:
    """True when the job title already says what the seniority filter says.

    One word in the request produces both. "AI SaaS founders" runs through
    _SENIORITY_RULES to seniority="founder" and through _TITLE_PHRASES to
    job_title="Founder" — the user stated one criterion and we send two, ANDed.

    That costs nothing only if the provider agrees with us about which
    seniority band a "Founder" sits in. Bytemine and Crustdata do not: both
    returned zero for every `title ~ "Founder" AND seniority = Owner` search in
    production, while Wiza — whose taxonomy happens to agree — returned real
    people for the identical query. The provider was not out of data; the second
    filter removed everyone the first one found.

    Dropping the implied one cannot widen the search past what was asked for:
    the title filter still carries the same word. It only stops us asserting a
    taxonomy the provider does not share.
    """
    if not job_title or not seniority:
        return False
    title = str(job_title).strip().lower()
    stated = _canonical_seniority(str(seniority))
    for pattern, key in _SENIORITY_RULES:
        if re.search(pattern, title):
            return _canonical_seniority(key) == stated
    return False


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


# =============================================================================
# GetLeads  (https://app.getleads.io)
# =============================================================================
#
# The first provider in the chain that can express a whole ICP as hard filters.
# Bytemine has no country field, Crustdata matches location as loose text, and
# ColdIQ has no industry or headcount field at all — which is why a search for
# "founders at 1-10 employee AI SaaS" came back with a Software Engineer at
# Google. /api/v1/contacts/search filters on industry, headcount band, seniority
# and geography together, so the stated segment is actually applied.
#
# It also has what nothing else in the chain has: a real `offset`. Repeat leads
# on ColdIQ have to be filtered out after the fact because its API cannot skip;
# here page two is page two.
#
# Cost: 1 credit per record *returned* by search, so this leg is billed like
# Wiza rather than like the reveal-on-unlock providers — it is capped to the
# page size and never over-fetches speculatively. `/search/count` is free, which
# is why the count is what gets logged rather than a second billed call.

GETLEADS_BASE = "https://app.getleads.io"

# How many pages a single search will walk looking for people it has not shown
# before. Each page is a round trip and is billed per record returned, so this
# is deliberately small: three depths of "everyone here is already seen" is a
# genuine no-new-results answer, not a reason to keep paying to page.
GETLEADS_MAX_PAGES = 3

# Their documented job-level enum: C-Team, VP, Director, Manager, Staff, Other.
# Only mappings the docs evidence are listed. "founder" and "owner" have no
# level of their own — the docs put founders under the CEO/Founder *persona*,
# and their own decision-maker lookup treats C-Team as the top level — so they
# map there. Anything unmapped is omitted rather than guessed: the job title
# filter still narrows the search.
_GL_SENIORITY = {
    "cxo": "C-Team", "c-level": "C-Team", "c-suite": "C-Team",
    "c_suite": "C-Team", "executive": "C-Team", "chief": "C-Team",
    "founder": "C-Team", "owner": "C-Team",
    "vp": "VP", "vice president": "VP", "svp": "VP", "evp": "VP",
    "director": "Director",
    "manager": "Manager", "head": "Manager",
}

# Their `regions` enum, and the continents they accept separately. This is the
# only provider in the chain with a field for either, which is what makes a
# search for "Europe" answerable here instead of merely refused.
_GL_REGION = {
    "emea": "EMEA", "apac": "APAC", "asia pacific": "APAC",
    "asia-pacific": "APAC", "anz": "APAC", "australasia": "APAC",
    "latam": "LATAM", "latin america": "LATAM",
    "noram": "NORAM", "north america": "NORAM", "americas": "NORAM",
}
_GL_CONTINENT = {
    "europe": "Europe", "asia": "Asia", "africa": "Africa",
    "south america": "South America", "oceania": "Oceania",
    "antarctica": "Antarctica",
}


def build_getleads_filters(p: dict) -> dict:
    """Translate internal search params into a GetLeads `filters` object.

    Almost nothing is refused here, because almost everything is expressible —
    which is the point of adding it. Industry reuses the LinkedIn spellings
    Bytemine needs (`_BM_INDUSTRY`): GetLeads documents its `industries` filter
    as LinkedIn's 441 categories, the same vocabulary.
    """
    f: dict = {}

    if p.get("job_title"):
        f["job_titles"] = [p["job_title"]]
    if p.get("seniority") and not seniority_implied_by_title(
            p.get("job_title"), p["seniority"]):
        mapped = _GL_SENIORITY.get(str(p["seniority"]).strip().lower())
        if mapped:
            f["seniority"] = [mapped]
    if p.get("departments"):
        depts = p["departments"]
        f["job_functions"] = depts if isinstance(depts, list) else [depts]

    if p.get("industry"):
        # Their validator names the vocabulary in its own 400: "Valid values: IT
        # Services and IT Consulting, Hospitals and Health Care, ...". That is
        # current LinkedIn, not the classic spelling Bytemine takes, and the
        # difference is not cosmetic — "Food & Beverages" is a 400 that drops
        # this leg out of the search.
        mapped = modern_linkedin_industry(p["industry"])
        if not mapped:
            raise ProviderUnsupported("industry", p["industry"])
        f["industries"] = [mapped]

    # Numeric bounds rather than the band labels: the server maps a range onto
    # its headcount bands, so we do not have to guess how it spells "1 to 10".
    if p.get("company_size"):
        lo, hi = _size_bounds(p["company_size"])
        if lo is None and hi is None:
            raise ProviderUnsupported("company_size", p["company_size"])
        if lo is not None:
            f["employees_min"] = lo
        if hi is not None:
            f["employees_max"] = hi

    domain = p.get("company_domain")
    company = p.get("company")
    if company and not domain and looks_like_domain(company):
        domain, company = company, None
    if domain:
        f["domains"] = [domain_host(domain)]
    elif company:
        f["company_name"] = company

    # A segment phrase belongs against the company's own description of itself,
    # which is where "AI SaaS" is actually written down.
    #
    # semantic_keywords are the same kind of phrase recovered from the user's
    # sentence when the frontend's parser folded it into an industry and dropped
    # the rest. This leg is usually the one that actually returns rows, so
    # without them a search for "B2B AI SaaS founders" was answered with any
    # founder at any small software company anywhere.
    segment = p.get("keywords") or p.get("semantic_keywords")
    if segment:
        f["company_description"] = str(segment)

    location = p.get("location") or p.get("company_location")
    if location:
        token = str(location)
        lowered = token.strip().lower()
        kind, value = classify_location(token)
        if kind == "region":
            if lowered in _GL_REGION:
                f["regions"] = [_GL_REGION[lowered]]
            elif lowered in _GL_CONTINENT:
                f["continents"] = [_GL_CONTINENT[lowered]]
            else:
                raise ProviderUnsupported("location", location)
        elif kind == "country":
            # Their filter takes country *names*; classify_location hands back
            # an ISO-2 code, so it is turned back into the name it came from.
            f["countries"] = [_GL_COUNTRY_NAME.get(value, token)]
        elif kind == "state":
            f["states"] = [_US_STATE_CODE_TO_NAME.get(value, value)]
        elif kind == "city":
            f["cities"] = [token]
        else:
            raise ProviderUnsupported("location", location)

    if not f:
        raise HTTPException(status_code=400, detail="At least one search parameter required")
    return f


def transform_getleads_contact(record: dict) -> dict:
    """Map one GetLeads contact row onto our lead shape.

    Their contact rows come from the same index the enrichment endpoints read,
    so the documented field names (first_name, last_name, email_address,
    cellphone, domain_org, org_company_name, person_country_name) are the ones
    read first; the camelCase convenience keys their responses also carry are
    accepted as fallbacks.
    """
    r = record or {}

    def pick(*names):
        for name in names:
            value = r.get(name)
            if value not in (None, ""):
                return value
        return None

    first = pick("first_name", "firstName") or ""
    last = pick("last_name", "lastName") or ""
    name = pick("full_name", "fullName", "name") or f"{first} {last}".strip()
    email = pick("email_address", "emailAddress", "email")
    phone, phone_type = pick_phone(r)
    if not phone:
        phone = clean_phone(pick("cellphone", "cellPhone", "phone_number"))
        phone_type = "mobile" if phone else None

    return {
        "contact_name": name or None,
        "first_name": first or None,
        "last_name": last or None,
        "job_title": pick("title", "job_title", "jobTitle", "position"),
        "company_name": pick("org_company_name", "company_name", "companyName",
                             "organization"),
        "company_domain": domain_host(
            pick("domain_org", "company_domain", "domain", "website") or "") or None,
        "business_email": email,
        # Their own verification verdict, carried through so the reveal-time
        # check can tell "already known bad" from "not yet checked".
        "email_status": pick("email_status", "emailStatus"),
        "phone": phone,
        "phone_type": phone_type,
        "phone_available": bool(phone),
        "linkedin_url": pick("person_linkedin_url", "linkedin_url", "linkedinUrl",
                             "profileUrl"),
        "industry": pick("industry", "org_industry", "company_industry"),
        "country": pick("person_country_name", "country", "countryName"),
        "provider": "getleads",
    }


def transform_treg_person(record: dict, search_params: dict = None) -> dict:
    """Normalize Treg's routed people output without depending on its child.

    Routed endpoints return a stable top-level person shape, but `raw` remains
    provider-native. The fallbacks below tolerate both so adding a new child to
    Treg does not require another provider implementation here.
    """
    r = record or {}
    search_params = search_params or {}

    def pick(*names):
        for name in names:
            value = r.get(name)
            if value not in (None, "", [], {}):
                return value
        return None

    first = pick("first_name", "firstName")
    last = pick("last_name", "lastName")
    name = pick("full_name", "fullName", "name") or " ".join(
        part for part in (first, last) if part)
    company = pick("company", "company_name", "companyName", "organization")
    if isinstance(company, dict):
        company_name = company.get("name")
        company_domain = company.get("domain") or company.get("website")
        company_industry = company.get("industry")
        company_headcount = company.get("employeeCount") or company.get("employee_count")
    else:
        company_name = company
        company_domain = pick("company_domain", "companyDomain", "domain", "website")
        company_industry = None
        company_headcount = None
    phone, phone_type = pick_phone(r)
    location = pick("location", "country")
    if isinstance(location, dict):
        location = ", ".join(
            str(location.get(key)) for key in ("city", "state", "country")
            if location.get(key)) or None

    return {
        "contact_name": name or None,
        "first_name": first,
        "last_name": last,
        "job_title": pick("title", "job_title", "jobTitle", "position"),
        "linkedin_url": pick("linkedin_url", "linkedinUrl", "linkedin", "profile_url"),
        "business_email": pick("email", "business_email", "work_email"),
        "email_status": pick("email_status", "emailStatus"),
        "phone": phone,
        "phone_type": phone_type,
        "location": location,
        "company_name": company_name,
        "company_domain": domain_host(company_domain or "") or None,
        "industry": (pick("industry", "company_industry") or company_industry
                     or search_params.get("industry")),
        "company_size": (pick("company_size", "headcount")
                         or (_size_bucket(company_headcount) if company_headcount else None)
                         or search_params.get("company_size")),
        "company_headcount": company_headcount,
        "company_revenue": pick("revenue", "company_revenue"),
        "company_funding": pick("funding", "company_funding"),
        "technologies": pick("technologies") or search_params.get("technologies"),
        "provider": "treg",
        "raw_data": r,
    }


def build_treg_people_search(params: dict, limit: int) -> dict:
    """Map the filters supported by Treg's routed lead-search capability."""
    unsupported = [
        field for field in (
            "departments", "company_size", "technologies", "intent_topics",
            "revenue_min", "revenue_max", "job_change_days", "signals",
        ) if params.get(field)
    ]
    if unsupported:
        raise ProviderUnsupported(", ".join(unsupported), "requested")

    location = params.get("location") or params.get("company_location")
    location_kind, normalized_location = classify_location(location) if location else (None, None)
    payload = {
        "q": params.get("query") or params.get("company"),
        "company_domain": params.get("company_domain"),
        "title": params.get("job_title"),
        "country": normalized_location if location_kind == "country" else None,
        "location": normalized_location if location_kind != "country" else None,
        "limit": max(min(int(limit or 10), 100), 1),
    }
    keywords = []
    if params.get("keywords"):
        keywords.extend(x.strip() for x in str(params["keywords"]).split(",") if x.strip())
    if params.get("industry"):
        keywords.append(str(params["industry"]))
    if params.get("seniority"):
        keywords.append(str(params["seniority"]))
    if keywords:
        payload["keywords"] = list(dict.fromkeys(keywords))
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def build_treg_leadsforge_search(params: dict, limit: int) -> dict:
    """Build an exact firmographic search for Treg's LeadsForge endpoint."""
    unsupported = [
        field for field in (
            "intent_topics", "revenue_min", "revenue_max", "job_change_days", "signals",
        ) if params.get(field)
    ]
    if unsupported:
        raise ProviderUnsupported(", ".join(unsupported), "requested")

    payload: dict = {"limit": max(min(int(limit or 10), 100), 1)}
    if params.get("company_domain"):
        payload["companyDomains"] = {"include": [domain_host(params["company_domain"])]}
    elif params.get("company"):
        payload["companyNames"] = {"include": [params["company"]]}
    if params.get("company_size"):
        low, high = _size_bounds(params["company_size"])
        if low is None and high is None:
            raise ProviderUnsupported("company_size", params["company_size"])
        payload["companyEmployeeNumberRange"] = {
            key: value for key, value in (("min", low), ("max", high))
            if value is not None
        }
    if params.get("industry"):
        payload["companyIndustries"] = {"include": [str(params["industry"]).lower()]}
    if params.get("technologies"):
        payload["companyTechnologies"] = {"any": params["technologies"]}
    if params.get("keywords"):
        payload["companyKeywords"] = {"include": [params["keywords"]]}
    if params.get("job_title"):
        payload["leadJobTitles"] = {"include": [params["job_title"]]}
    if params.get("seniority"):
        payload["leadSeniorities"] = {"include": [_canonical_seniority(params["seniority"])]}
    if params.get("departments"):
        payload["leadDepartments"] = {"include": params["departments"]}
    if params.get("location"):
        payload["leadLocations"] = {"include": [params["location"]]}
    if params.get("company_location"):
        payload["companyLocations"] = {"include": [params["company_location"]]}
    payload["companyRequired"] = True
    return payload


def treg_search_plan(params: dict, limit: int) -> tuple[str, dict]:
    """Choose the Treg lead endpoint that can express the requested ICP."""
    needs_firmographics = any(
        params.get(field) for field in ("company_size", "technologies", "departments"))
    if needs_firmographics:
        return "leadsforge.people.search", build_treg_leadsforge_search(params, limit)
    return "treg.people.search", build_treg_people_search(params, limit)


async def treg_person_search(params: dict, limit: int, cursor: str = None) -> dict:
    endpoint_id, payload = treg_search_plan(params, limit)
    data = await treg_call(
        endpoint_id, payload, "lead-search", query={"cursor": cursor} if cursor else None)
    if endpoint_id == "leadsforge.people.search":
        people = data.get("leads") or [] if isinstance(data, dict) else []
        total = len(people)
        next_cursor = data.get("cursor") if isinstance(data, dict) else None
    else:
        output = data.get("output") if isinstance(data, dict) else None
        output = output if isinstance(output, dict) else {}
        people = output.get("people") or []
        total = output.get("total") or len(people)
        next_cursor = output.get("next_cursor")
    if not isinstance(people, list):
        people = []
    return {
        "profiles": people,
        "total": total,
        "next_cursor": next_cursor,
    }


async def getleads_call(path: str, body: dict) -> dict:
    """POST to GetLeads and return the parsed body, mapping their errors onto ours."""
    headers = {
        "Authorization": f"Bearer {settings.getleads_api_key or ''}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(f"{GETLEADS_BASE}{path}", headers=headers, json=body)

    print(f"GetLeads {path} status: {resp.status_code} {resp.text[:300]}")
    if resp.status_code == 401:
        raise HTTPException(status_code=502, detail="GetLeads rejected the API key")
    if resp.status_code == 402:
        raise HTTPException(
            status_code=402,
            detail="GetLeads credits exhausted — top up to keep searching")
    if resp.status_code == 429:
        raise HTTPException(status_code=429,
                            detail="GetLeads rate limit — try again in a moment")
    if resp.status_code not in (200, 201, 202):
        raise HTTPException(
            status_code=resp.status_code if resp.status_code < 500 else 502,
            detail=f"GetLeads error: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="GetLeads returned a non-JSON body")


async def getleads_person_search(params: dict, limit: int, offset: int = 0) -> dict:
    """Search the GetLeads contact index.

    Billed per record returned, so `limit` is the page size and nothing more:
    this leg never over-fetches to compensate for filtering, because unlike
    ColdIQ it does not have to — `offset` moves to genuinely new rows.
    """
    body = {
        "filters": build_getleads_filters(params),
        "limit": max(min(limit, 50000), 1),
        "offset": max(offset, 0),
        # Without this one prolific company can fill a whole page.
        "max_per_company": 3,
    }
    print(f"GetLeads search body: {json.dumps(body)[:400]}")
    data = await getleads_call("/api/v1/contacts/search", body)

    contacts = data.get("contacts") or data.get("results") or data.get("data") or []
    contacts = [c for c in contacts if isinstance(c, dict)]
    if contacts:
        print(f"GetLeads record keys: {sorted(contacts[0].keys())[:25]}")

    return {
        "profiles": contacts,
        "total": data.get("total_available") or len(contacts),
        "next_offset": data.get("next_offset") if data.get("has_more") else None,
    }


async def fetch_from_getleads(params: dict) -> list:
    """People-search fetcher for /search — returns raw GetLeads contact rows."""
    limit = max(min(params.get("limit", 10), 100), 1)
    data = await getleads_person_search(params, limit)
    return data["profiles"]


# =============================================================================
# Findymail  (https://app.findymail.com)
# =============================================================================
#
# Three different jobs, and the useful ones are not the search.
#
# Its synchronous endpoints answer two problems this chain has right now.
# /api/search/business-profile turns a LinkedIn URL into a verified email —
# which is exactly what a ColdIQ lead is missing, since that provider can
# return a person as nothing but a profile URL and a headline. And /api/verify
# is a second deliverability checker, which matters because ColdIQ is both the
# only verifier and the thing that runs out of credits.
#
# Both are charged only on success ("uses one finder credit if a verified email
# is found"), so a miss costs nothing and they can be tried freely.
#
# Its search — Intellimatch — is asynchronous: submit a natural-language query,
# get a hash, poll for completion, then page the results. That is a poor fit for
# a request a user is waiting on, so the leg is bounded (see FINDYMAIL_POLL_*)
# and sits late in the waterfall.

FINDYMAIL_BASE = "https://app.findymail.com"

# How long a live search will wait for an Intellimatch task. Findymail queues
# these; a large export can take minutes, which no one is going to sit through.
# On a timeout the hash is logged so the same task can be collected later rather
# than re-run — the credits are already spent either way.
FINDYMAIL_POLL_SECONDS = 25.0
FINDYMAIL_POLL_INTERVAL = 2.5


async def findymail_call(method: str, path: str, *, json_body: dict = None,
                         params: dict = None) -> tuple:
    """Call Findymail. Returns (status_code, parsed body or None).

    Returns rather than raises for the outcomes that are ordinary here: 402 out
    of credits and 423 subscription paused are states of their account, not
    failures of this request, and a leg that raises on them takes the whole
    search down with it.
    """
    headers = {
        "Authorization": f"Bearer {settings.findymail_api_key or ''}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                method, f"{FINDYMAIL_BASE}{path}",
                headers=headers, json=json_body, params=params)
    except httpx.HTTPError as exc:
        print(f"Findymail {path} unreachable: {exc}")
        return 0, None

    if resp.status_code != 200:
        print(f"Findymail {path} status: {resp.status_code} {resp.text[:200]}")
    try:
        return resp.status_code, resp.json()
    except ValueError:
        # /api/verify documents a text/plain body holding JSON.
        try:
            return resp.status_code, json.loads(resp.text)
        except ValueError:
            return resp.status_code, None


def findymail_domain_for(domain: str = None, company: str = None) -> Optional[str]:
    """A real mail domain for /api/search/name, or None when we only have a name.

    /api/search/name matches a person against a company's mail domain, so it
    needs an actual domain. Falling back to the company name sent it strings
    like "Uncapped", which cannot resolve: the leg was a guaranteed miss on
    every lead we knew only by name, which is most of them. Worse than wasted —
    a garbage domain that did match would hand back somebody else's address.
    """
    for candidate in (domain, company):
        if candidate and looks_like_domain(candidate):
            return domain_host(candidate) or candidate
    return None


async def findymail_find_email(*, linkedin_url: str = None, name: str = None,
                               domain: str = None) -> dict:
    """Find one verified email. Returns {} when nothing was found.

    A LinkedIn URL is tried first because it identifies a person exactly; name
    plus domain can land on the wrong person at a big company. Charged only when
    an email is actually found, so trying both costs nothing on a miss.
    """
    if linkedin_url:
        status, body = await findymail_call(
            "POST", "/api/search/business-profile",
            json_body={"linkedin_url": linkedin_url})
        contact = (body or {}).get("contact") if status == 200 else None
        if contact and contact.get("email"):
            return contact

    if name and domain:
        status, body = await findymail_call(
            "POST", "/api/search/name",
            json_body={"name": name, "domain": domain_host(domain) or domain})
        contact = (body or {}).get("contact") if status == 200 else None
        if contact and contact.get("email"):
            return contact

    return {}


async def findymail_verify_email(email: str) -> dict:
    """Verify one address through Findymail, in our shared verdict shape.

    Its answer is a bare boolean — `verified` — with none of the catch-all or
    role detail ColdIQ returns, so a true maps to "deliverable" and a false to
    "undeliverable" and nothing is invented in between.
    """
    status, body = await findymail_call("POST", "/api/verify",
                                        json_body={"email": email})
    if status != 200 or not isinstance(body, dict) or "verified" not in body:
        reason = ("findymail out of verifier credits" if status == 402
                  else "findymail subscription paused" if status == 423
                  else "findymail returned no verdict")
        return {"status": "unknown", "sendable": None,
                "checked_by": "findymail", "reason": reason}

    deliverable = bool(body.get("verified"))
    return {
        "status": "deliverable" if deliverable else "undeliverable",
        "sendable": deliverable,
        "checked_by": "findymail",
        "vendor": body.get("provider"),
        "score": None,
        "raw_status": str(body.get("verified")),
    }


# A pitch says what the user sells. An ICP says who the buyers are. Intellimatch
# searches company descriptions, so only the second is a query it can answer.
_PITCH_MARKERS = re.compile(
    r"\b(i am selling|i'm selling|i sell|we are selling|we're selling|we sell|"
    r"i offer|we offer|i charge|we charge|i build|we build|i provide|we provide|"
    r"i help|we help|my product|our product|my service|our service|my website|"
    r"our website|it cost|it costs|per month)\b", re.I)

# Past this, a "description of target companies" is something else.
FINDYMAIL_MAX_QUERY_CHARS = 300


def findymail_query_from_sentence(sentence: str) -> str:
    """The user's sentence, when it describes the companies to find.

    Production sent this endpoint entire sales pitches — "I am selling a website
    to car showrooms, https://... it cost only 50$ ... for maintaince I charge
    5dolllar per month ..." — several hundred words of offer, pricing and links.
    Intellimatch answered `failed` to every one of them, on every search, because
    none of it describes a company to go and find.

    Returned empty when the sentence is a pitch, which sends the caller to the
    structured filters instead. Those describe the buyer, which is the question
    this endpoint takes.
    """
    text = re.sub(r"https?://\S+", " ", sentence or "").strip()
    if not text or _PITCH_MARKERS.search(text) or len(text) > FINDYMAIL_MAX_QUERY_CHARS:
        return ""
    return " ".join(text.split())


def findymail_intellimatch_query(p: dict) -> str:
    """The natural-language query Intellimatch searches companies with.

    It takes a sentence, not filters, so the user's own sentence is used when it
    reads like a description of the companies to find. A sales pitch is not
    that, so the structured filters are written back out as a sentence instead —
    an empty query is rejected, and a bare job title describes a person rather
    than a company.
    """
    sentence = findymail_query_from_sentence(
        p.get("semantic_query") or p.get("query") or "")
    if sentence:
        return sentence

    described = []
    if p.get("industry"):
        described.append(str(p["industry"]))
    size = p.get("company_size")
    location = p.get("company_location") or p.get("location")
    keywords = p.get("keywords") or p.get("semantic_keywords")

    # "companies" on its own describes every company there is. A search with a
    # job title and nothing else says who to find, never where — and Intellimatch
    # searches companies, so that has to be refused rather than sent as a query
    # that matches the entire index.
    if not (described or size or location or keywords):
        return ""

    parts = described + ["companies"]
    if size:
        parts.append(f"with {size} employees")
    if location:
        parts.append(f"in {location}")
    if keywords:
        parts.append(f"({keywords})")
    return " ".join(parts).strip()


def build_findymail_search(p: dict, limit: int) -> dict:
    """Build the Intellimatch task body.

    require_email is on deliberately. Findymail documents that companies without
    an email are "excluded and not charged", which turns the cost of a bad
    search into nothing and means this leg never returns the emailless rows that
    the rest of the chain already produces too many of.
    """
    query = findymail_intellimatch_query(p)
    if not query:
        raise ProviderUnsupported("query", "no company description to search")

    config = {
        "find_contact": True,
        "find_email": True,
        "require_email": True,
        "mode": "targeted" if p.get("job_title") else "broad",
    }
    if p.get("job_title"):
        # Tiers are fallbacks, not alternatives: tier two is only tried when
        # tier one finds nobody, so the stated title stays the first choice.
        config["target_job_titles"] = [[p["job_title"]]]

    return {"query": query, "limit": max(min(limit, 5000), 1), "config": config}


async def findymail_person_search(params: dict, limit: int) -> dict:
    """Run one Intellimatch search, within a bounded wait.

    Submit, poll, page. A task that has not finished inside the budget returns
    what it has rather than nothing, and logs its hash: the credits are spent
    when the task runs, so the results stay collectable afterwards.
    """
    body = build_findymail_search(params, limit)
    print(f"Findymail intellimatch body: {json.dumps(body)[:400]}")

    status, submitted = await findymail_call(
        "POST", "/api/intellimatch/search", json_body=body)
    task = (submitted or {}).get("hash") if status == 200 else None
    if not task:
        if status == 402:
            raise HTTPException(
                status_code=402,
                detail="Findymail credits exhausted — top up to keep searching")
        if status == 423:
            raise HTTPException(status_code=502,
                                detail="Findymail subscription is paused")
        raise HTTPException(status_code=502, detail="Findymail started no search")

    deadline = time.monotonic() + FINDYMAIL_POLL_SECONDS
    state = "pending"
    while time.monotonic() < deadline:
        await asyncio.sleep(FINDYMAIL_POLL_INTERVAL)
        _, progress = await findymail_call(
            "GET", "/api/intellimatch/status", params={"hash": task})
        state = (progress or {}).get("status") or state
        if state in ("success", "failed", "not_found"):
            break
        pct = (progress or {}).get("progress")
        print(f"Findymail {task[:12]}… {state}"
              + (f" {pct}%" if pct not in (None, "") else ""))

    if state in ("failed", "not_found"):
        print(f"Findymail search {state}: {task}")
        return {"profiles": [], "total": 0, "next_cursor": None}

    _, page = await findymail_call(
        "GET", "/api/intellimatch/data",
        params={"hash": task, "page": 1, "per_page": max(min(limit, 500), 1)})
    rows = [r for r in ((page or {}).get("data") or []) if isinstance(r, dict)]

    if state != "success":
        # Partial, not empty: the task keeps running on their side and this hash
        # can be collected later without paying for the search twice.
        print(f"Findymail still {state} after {FINDYMAIL_POLL_SECONDS:.0f}s — "
              f"returning {len(rows)} row(s) so far, hash {task}")
    if rows:
        print(f"Findymail record keys: {sorted(rows[0].keys())[:20]}")

    return {
        "profiles": rows,
        "total": ((page or {}).get("meta") or {}).get("total") or len(rows),
        "next_cursor": None,
    }


def transform_findymail_row(row: dict, search_params: dict = None) -> dict:
    """Map one Intellimatch result onto our lead shape.

    Each row is a company with one contact folded into it — the contact_* keys
    are the person, everything else describes where they work.
    """
    r = row or {}
    industries = r.get("industries")
    if isinstance(industries, list):
        industry = industries[0] if industries else None
    else:
        industry = industries or None

    phone, phone_type = pick_phone(r)
    if not phone:
        phone = clean_phone(r.get("contact_phone"))
        phone_type = None

    name = (r.get("contact_name") or "").strip()
    first, _, last = name.partition(" ")

    return {
        "contact_name": name or None,
        "first_name": first or None,
        "last_name": last or None,
        "job_title": r.get("contact_job_title"),
        "company_name": r.get("name"),
        "company_domain": domain_host(r.get("domain") or "") or None,
        "company_description": r.get("description"),
        "business_email": r.get("contact_email"),
        "phone": phone,
        "phone_type": phone_type,
        "phone_available": bool(phone),
        "linkedin_url": r.get("contact_linkedin_url"),
        "industry": industry or (search_params or {}).get("industry"),
        "company_size": r.get("employee_count_range"),
        "country": r.get("country"),
        # Intellimatch's own confidence that the company fits the query. Carried
        # through rather than folded into our score: it measures a different
        # thing, and one provider's number should not silently become ours.
        "match_score": r.get("match_score"),
        "provider": "findymail",
    }


async def fetch_from_findymail(params: dict) -> list:
    """People-search fetcher for /search — returns raw Intellimatch rows."""
    limit = max(min(params.get("limit", 10), 500), 1)
    data = await findymail_person_search(params, limit)
    return data["profiles"]


# =============================================================================
# Fiber AI  (https://api.fiber.ai)
# =============================================================================
#
# The first provider in the chain that can express this product's whole ICP —
# job title, industry, country, and headcount — against LinkedIn's own
# vocabulary, with a real pagination cursor rather than an offset.
#
# It does all three jobs the chain needs, so it is wired into all three places:
# people search, a reveal keyed on a LinkedIn URL, and email validation. The
# validator matters most in the short term: ColdIQ is out of credits and
# Findymail answers a bare boolean, while Fiber returns the catch-all,
# role-based and disposable detail ColdIQ used to be the only source of.
#
# Cost: 1 credit per profile returned by search, 2 per revealed work email, 1
# per validation. Nothing is charged for a search that matches nothing.

FIBER_BASE = "https://api.fiber.ai"
FIBER_MAX_PAGE = 1000

# Fiber and GetLeads both name industries the way *current* LinkedIn does, which
# is not the classic vocabulary the rest of this file holds — see
# linkedin_industry. These are the eight of ours spelled differently there; the
# rest pass through unchanged.
#
# An industry outside the vocabulary is refused rather than sent. This is the
# Bytemine lesson: an unmapped industry name does not error, it silently
# matches nothing, and the chain reads that as "this provider has no such
# people" instead of "we asked the wrong question". GetLeads is not even that
# quiet about it — it 400s with "Invalid industries: Food & Beverages. Valid
# values: IT Services and IT Consulting, Hospitals and Health Care, ..." and
# drops out of the search as an error.
_MODERN_INDUSTRY_ALIASES = {
    "Computer & Network Security": "Computer and Network Security",
    "Education": "Education Management",
    "Food & Beverages": "Food and Beverage Services",
    "Health Care": "Hospitals and Health Care",
    "Hospital & Health Care": "Hospitals and Health Care",
    "Non-Profit Organization Management": "Non-Profit Organizations",
    "Oil & Energy": "Oil and Gas",
    "Pharmaceuticals": "Pharmaceutical Manufacturing",
}

# The industries we can produce that the modern vocabulary spells our way.
_MODERN_INDUSTRY_NATIVE = frozenset({
    "Automotive", "Banking", "Biotechnology", "Computer Games",
    "Computer Software", "Construction", "E-Learning", "Financial Services",
    "Government Administration", "Hospitality",
    "Information Technology and Services", "Insurance", "Internet",
    "Legal Services", "Logistics and Supply Chain", "Management Consulting",
    "Manufacturing", "Marketing and Advertising", "Media Production",
    "Real Estate", "Retail", "Staffing and Recruiting", "Telecommunications",
})

# Their country filter takes ISO-3; classify_location hands back ISO-2. This
# covers every code our own parser can emit — FiberCountryCoverageTests holds
# that true — so the refusal below is a guard against the map falling behind
# _BM_COUNTRY_CODE, not an expected outcome.
_FIBER_COUNTRY3 = {
    "AE": "ARE", "AR": "ARG", "AT": "AUT", "AU": "AUS", "BE": "BEL",
    "BR": "BRA", "CA": "CAN", "CH": "CHE", "CL": "CHL", "CN": "CHN",
    "CO": "COL", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "HU": "HUN", "IE": "IRL",
    "IL": "ISR", "IN": "IND", "IT": "ITA", "JP": "JPN", "KE": "KEN",
    "KR": "KOR", "MX": "MEX", "NG": "NGA", "NL": "NLD", "NO": "NOR",
    "NZ": "NZL", "PL": "POL", "PT": "PRT", "RO": "ROU", "RU": "RUS",
    "SA": "SAU", "SE": "SWE", "SG": "SGP", "TR": "TUR", "TW": "TWN",
    "UA": "UKR", "US": "USA", "ZA": "ZAF",
}

# free-form-city requires a radius, so one has to be chosen. A city in an ICP
# means the metro, not the boundary line — nobody asking for founders in San
# Francisco means to exclude Oakland — and this is the radius their own city
# presets are built around.
FIBER_CITY_RADIUS_MILES = 25


def modern_linkedin_industry(value: str) -> Optional[str]:
    """Current LinkedIn's spelling of an industry, or None when it has none.

    Used by every provider on the newer taxonomy — Fiber and GetLeads today.
    Bytemine is still on the classic names, which is what linkedin_industry
    returns.
    """
    mapped = linkedin_industry(value)
    if not mapped:
        return None
    if mapped in _MODERN_INDUSTRY_ALIASES:
        return _MODERN_INDUSTRY_ALIASES[mapped]
    return mapped if mapped in _MODERN_INDUSTRY_NATIVE else None


def fiber_industry(value: str) -> Optional[str]:
    """Fiber's spelling of an industry, or None when it is not one of theirs."""
    return modern_linkedin_industry(value)


def industry_is_expressible(value: str) -> bool:
    """True when some provider in the chain has a field for this industry."""
    return bool(linkedin_industry(value) or modern_linkedin_industry(value))


async def fiber_call(path: str, body: dict) -> tuple:
    """Call Fiber. Returns (status_code, parsed body or None).

    The key goes in the `x-api-key` header even though Fiber also accepts it in
    the request body. Every provider module here logs its request body, and a
    key in the body is a key in the Railway logs.

    Returns rather than raises on 402 and 429: those are states of the account,
    not failures of this request, and a leg that raises on them takes the whole
    search down with it.
    """
    headers = {
        "x-api-key": settings.fiber_api_key or "",
        "Content-Type": "application/json",
    }
    print(f"Fiber {path} body: {json.dumps(body)[:400]}")
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{FIBER_BASE}{path}", headers=headers, json=body)
    except httpx.HTTPError as exc:
        print(f"Fiber {path} unreachable: {exc}")
        return 0, None

    if resp.status_code != 200:
        print(f"Fiber {path} status: {resp.status_code} {resp.text[:300]}")
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, None


def build_fiber_people_params(p: dict) -> dict:
    """The person-level half of a Fiber search.

    Raises ProviderUnsupported for a filter their people index has no field for,
    so the chain reaches a provider that does rather than searching without it.
    """
    sp: dict = {}

    if p.get("job_title"):
        sp["jobTitleV3"] = {"anyOf": [{"type": "plain", "term": p["job_title"]}]}

    if p.get("industry"):
        mapped = fiber_industry(p["industry"])
        if not mapped:
            raise ProviderUnsupported("industry", p["industry"])
        sp["industry"] = {"anyOf": [mapped]}

    location = p.get("location") or p.get("company_location")
    if location:
        kind, value = classify_location(str(location))
        if kind == "country":
            code = _FIBER_COUNTRY3.get(value)
            if not code:
                raise ProviderUnsupported("location", location)
            sp["country3LetterCode"] = {"anyOf": [code]}
        elif kind == "state":
            sp["state"] = {"anyOf": [{
                "stateName": _US_STATE_CODE_TO_NAME.get(value, value),
                "countryCode": "USA",
            }]}
        elif kind == "city":
            sp["location"] = {"unionAll": [{
                "strategy": "free-form-city",
                "city": str(location),
                "radius": {"unit": "miles", "quantity": FIBER_CITY_RADIUS_MILES},
            }]}
        else:
            # A continent or multi-country region. Their presets are metro
            # areas, not continents, so there is nothing here to express it.
            raise ProviderUnsupported("location", location)

    # A segment phrase belongs in free text. keywordsV2 searches headlines,
    # summaries and current job titles together, which is where "AI SaaS" is
    # actually written on a profile.
    segment = p.get("keywords") or p.get("semantic_keywords")
    if segment:
        terms = [t for t in str(segment).split(",") if t.strip()] or [str(segment)]
        sp["keywordsV2"] = {
            "operator": "AND",
            "clauses": [{"operator": "OR", "terms": [t.strip() for t in terms]}],
        }

    if not sp:
        raise HTTPException(status_code=400, detail="At least one search parameter required")
    return sp


def build_fiber_search(p: dict, limit: int, cursor: str = None) -> tuple:
    """Return (path, body) for the endpoint that can express this ICP.

    /v1/people-search has no headcount field. Refusing on company_size would be
    honest and would also make Fiber sit out nearly every search this product
    sends, the way Bytemine sits out every country-level one — so an ICP with a
    size goes to the combined endpoint, which filters companies and people
    together. It costs a company credit on top of the profile credit, which is
    the price of expressing the whole ICP rather than most of it.
    """
    people = build_fiber_people_params(p)
    page = max(min(limit, FIBER_MAX_PAGE), 1)

    if not p.get("company_size"):
        body = {"searchParams": people, "pageSize": page, "includeCount": False}
        if cursor:
            body["cursor"] = cursor
        return "/v1/people-search", body

    lo, hi = _size_bounds(p["company_size"])
    if lo is None and hi is None:
        raise ProviderUnsupported("company_size", p["company_size"])

    company: dict = {}
    # lowerBoundExclusive is exclusive: a 1-10 band starts above 0, not above 1.
    if lo is not None:
        company["lowerBoundExclusive"] = max(lo - 1, 0)
    if hi is not None:
        company["upperBoundInclusive"] = hi

    body = {
        "companyConfig": {"searchParams": {"employeeCountV2": company},
                          "pageSize": page},
        "profileConfig": {"searchParams": people, "pageSize": page},
    }
    if cursor:
        body["profileConfig"]["profileCursor"] = cursor
    return "/v1/combined-search/paginated", body


async def fiber_person_search(params: dict, limit: int, cursor: str = None) -> dict:
    """Search Fiber for people. Billed per profile returned, so never over-fetch."""
    path, body = build_fiber_search(params, limit, cursor)
    status, data = await fiber_call(path, body)
    if status != 200 or not isinstance(data, dict):
        return {"profiles": [], "total": 0, "next_cursor": None}

    output = data.get("output") or {}
    rows = output.get("profiles") if "profiles" in output else output.get("data")
    rows = rows or []
    # estimatedCount is only populated when includeCount is set, which costs a
    # credit — the row count is what was actually returned and paid for.
    next_cursor = output.get("nextProfilesCursor") or output.get("nextCursor")
    print(f"Fiber returned {len(rows)} profile(s)")
    return {"profiles": rows, "total": len(rows), "next_cursor": next_cursor}


def transform_fiber_profile(row: dict, search_params: dict = None) -> dict:
    """Map one Fiber profile onto our lead shape.

    Search returns no contact details — those are a separate, separately billed
    reveal — so email and phone are None here, exactly as for Bytemine and
    Crustdata.
    """
    del search_params  # their ranking orders the page; we do not re-score it
    job = row.get("current_job") or {}
    name = row.get("name")
    first, last = row.get("first_name"), row.get("last_name")
    if name and not (first or last):
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0], " ".join(parts[1:])
        elif parts:
            first = parts[0]

    slug = row.get("primary_slug")
    linkedin_url = row.get("url") or (
        f"https://www.linkedin.com/in/{slug}" if slug else None)

    return {
        "contact_name": name or " ".join(x for x in (first, last) if x) or None,
        "first_name": first,
        "last_name": last,
        "job_title": job.get("title") or row.get("headline"),
        "seniority": job.get("seniority"),
        "company_name": job.get("company_name"),
        "company_domain": None,
        "industry": row.get("industry_name"),
        "location": row.get("locality") or job.get("locality"),
        "linkedin_url": linkedin_url,
        "contact_linkedin_url": linkedin_url,
        "headline": row.get("headline"),
        "business_email": None,
        "email": None,
        "email_available": None,
        "phone": None,
        "phone_type": None,
        "phone_available": None,
        # No relevance score of our own: their ranking already ordered the
        # page, and relevance_score is theirs to explain, not ours to fold in.
        "score": row.get("relevance_score"),
        "provider": "fiber",
    }


async def fiber_reveal(linkedin_url: str, want_phone: bool = False) -> dict:
    """Reveal a work email (and optionally a phone) from a LinkedIn URL.

    Charged per requested type, so personal emails are never asked for: this is
    B2B outbound and a personal address is both dearer and less useful.
    """
    if not linkedin_url:
        return {}

    status, data = await fiber_call("/v1/contact-details/single", {
        "linkedinUrl": linkedin_url,
        "enrichmentType": {"getWorkEmails": True, "getPersonalEmails": False,
                           "getPhoneNumbers": bool(want_phone)},
        "patience": "MEDIUM",
    })
    if status != 200 or not isinstance(data, dict):
        return {}

    profile = ((data.get("output") or {}).get("profile")) or {}
    emails = profile.get("emails") or []
    # `valid` is the only status that has passed deliverability verification;
    # their own docs say to treat `unknown` cautiously. Prefer a verified one.
    best = next((e for e in emails if e.get("status") == "valid"), None) or (
        emails[0] if emails else None)
    phones = profile.get("phoneNumbers") or []

    return {
        "name": profile.get("name"),
        "email": (best or {}).get("email"),
        "email_status": (best or {}).get("status"),
        "phone": (phones[0] or {}).get("number") if phones else None,
        "phone_type": (phones[0] or {}).get("type") if phones else None,
    }


# Their verdict vocabulary, mapped onto ours. "risky" is a real answer — a
# catch-all domain that accepts everything — and it is not sendable, so it is
# kept distinct from "we could not tell".
_FIBER_VERDICT = {
    "ok": ("deliverable", True),
    "undeliverable": ("undeliverable", False),
    "risky": ("risky", False),
    "inconclusive": ("unknown", None),
}


async def fiber_verify_email(email: str) -> dict:
    """Verify one address through Fiber, in our shared verdict shape.

    Richer than Findymail's bare boolean: catch-all, role-based and disposable
    all come back, which is the detail ColdIQ was the only source of.
    """
    status, body = await fiber_call("/v1/validate-email/single", {"email": email})
    output = (body or {}).get("output") if isinstance(body, dict) else None
    if status != 200 or not isinstance(output, dict) or not output.get("verdict"):
        reason = ("fiber out of credits" if status == 402
                  else "fiber rate limited" if status == 429
                  else "fiber returned no verdict")
        return {"status": "unknown", "sendable": None,
                "checked_by": "fiber", "reason": reason}

    mapped, sendable = _FIBER_VERDICT.get(output["verdict"], ("unknown", None))
    return {
        "status": mapped,
        "sendable": sendable,
        "checked_by": "fiber",
        "raw_status": output["verdict"],
        "catch_all": output.get("is_catch_all"),
        "role_based": output.get("is_role_based"),
        "disposable": output.get("is_disposable"),
        "free_provider": output.get("is_consumer"),
        "vendor": output.get("email_provider"),
        "score": None,
    }


# =============================================================================
# ColdIQ  (https://api.coldiq.com)
# =============================================================================
#
# ColdIQ is a meta-provider: one API in front of Apollo, Prospeo, Wiza and
# others, with its own managed waterfall behind `provider: "auto"`. That means
# it can return the same person our own Wiza leg returns — the merge dedupes on
# LinkedIn URL, email, then name+company, so an overlap costs a duplicate call
# rather than a duplicate lead.
#
# Its people search is deliberately narrow: it finds decision-makers *at
# companies*, keyed on domain or LinkedIn company URL, plus title, seniority and
# location. There is no industry filter and no headcount filter on the input, so
# an ICP stated in those terms cannot be applied exactly — see
# build_coldiq_filters, which sends them as ranking hints and flags the rows.
#
# It has a second job here that the other three providers do not: /v1/email/verify
# is the deliverability check every revealed email goes through, whichever
# provider sourced it (see coldiq_verify_email and verify_revealed_lead).

COLDIQ_BASE = "https://api.coldiq.com"

# How far past the requested page to fetch, to leave room for already-seen rows
# to be filtered out. ColdIQ has no exclusion field, so the filtering happens
# here; it also bills per record returned, so this cannot be generous. Three
# pages' worth is the compromise — enough to clear a page of repeats, far short
# of the "one row per person ever seen" that made a 6-lead search ask for 56.
COLDIQ_OVERFETCH = 3

# ColdIQ's documented seniority vocabulary, from the FindPeopleInput examples
# ("c_suite", "vp"). Ours is mapped onto it; anything unmapped is left out
# rather than guessed, and the title filter still narrows the search.
_CIQ_SENIORITY = {
    "c_suite": "c_suite", "cxo": "c_suite", "c-level": "c_suite",
    "c-suite": "c_suite", "executive": "c_suite",
    "vp": "vp", "vice president": "vp",
    "director": "director",
    "manager": "manager",
    "senior": "senior",
    "entry": "entry", "junior": "entry",
    "owner": "owner", "founder": "owner",
    "partner": "partner",
}


def build_coldiq_filters(p: dict) -> dict:
    """Translate internal search params into a ColdIQ FindPeopleInput.

    Raises ProviderUnsupported for the filters its input has no field for.
    `industry` and `company_size` are the significant ones: an ICP that names a
    segment or a headcount band cannot be expressed here, and returning
    whoever matched the remaining filters would be answering a different
    question.
    """
    body: dict = {}

    if p.get("job_title"):
        body["job_titles"] = [p["job_title"]]
    if p.get("keywords"):
        body["keywords"] = [p["keywords"]]

    if p.get("seniority") and not seniority_implied_by_title(
            p.get("job_title"), p["seniority"]):
        mapped = _CIQ_SENIORITY.get(str(p["seniority"]).strip().lower())
        if mapped:
            body["seniorities"] = [mapped]

    # Domain is the filter ColdIQ is built around — it finds people *at* a
    # company. A bare company name is not accepted, so it is not sent.
    domain = p.get("company_domain")
    company = p.get("company")
    if company and not domain and looks_like_domain(company):
        domain, company = company, None
    if domain:
        body["company_domains"] = [domain_host(domain)]
    elif company:
        raise ProviderUnsupported("company", company)

    # `locations` takes ISO-2 codes or country names, for the *person*.
    location = p.get("location") or p.get("company_location")
    if location:
        kind, value = classify_location(str(location))
        if kind == "country":
            body["locations"] = [value]
        elif kind in ("state", "city"):
            # A sub-national place would be read as a country name and silently
            # match nothing; the whole point of the filter is to narrow.
            raise ProviderUnsupported("location", location)
        else:
            raise ProviderUnsupported("location", location)

    # Neither has a field on FindPeopleInput, but refusing outright made ColdIQ
    # inert: nearly every real search names an industry or a headcount band, so
    # it sat out every one and contributed nothing at all.
    #
    # They go in as keywords instead, which is a ranking hint to ColdIQ's own
    # provider waterfall rather than a hard filter. That is a real weakening —
    # a ColdIQ result is title/seniority/location-exact but only
    # segment-*probable* — so its rows are flagged (see transform_coldiq_profile)
    # and the caller can tell them apart from a row that matched a stated
    # industry outright.
    hints: list = []
    if p.get("industry"):
        hints.append(str(p["industry"]))
    if p.get("company_size"):
        hints.append(f"{p['company_size']} employees")
    if hints:
        body["keywords"] = list(dict.fromkeys((body.get("keywords") or []) + hints))
        body["_unverified_dimensions"] = [
            d for d in ("industry", "company_size") if p.get(d)
        ]

    # Titles, seniorities, locations and domains are the filters ColdIQ applies
    # exactly. With none of them there is nothing holding the search to the ICP
    # at all — keywords alone would return whoever the waterfall liked.
    anchored = any(body.get(k) for k in
                   ("job_titles", "seniorities", "locations", "company_domains"))
    if not anchored:
        raise ProviderUnsupported("industry" if p.get("industry") else "company_size",
                                  p.get("industry") or p.get("company_size"))

    if not body:
        raise HTTPException(status_code=400, detail="At least one search parameter required")
    return body


# Corporate suffixes that mark the tail of a headline as a company rather than
# more of the job title. "Owner, WitWerx, Inc." is a company; "Engineer, Senior"
# is not, and guessing on a bare comma would invent employers.
_COMPANY_SUFFIX = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|ag|bv|nv|sa|srl|spa|plc|"
    r"pty|llp|lp|group|holdings|labs|studio|studios|agency|partners)\b\.?$",
    re.IGNORECASE,
)


def name_from_linkedin_url(url: str) -> Optional[str]:
    """Recover a display name from a LinkedIn profile slug.

    "/in/seb-hall" is "Seb Hall"; "/in/mark-lucovsky-5280034" is Mark Lucovsky
    with LinkedIn's disambiguating id on the end, which is dropped. A slug that
    is one unbroken token ("juliendanjou") cannot be split into a name and is
    left alone rather than mangled.
    """
    if not url:
        return None
    match = re.search(r"/(?:in|pub)/([^/?#]+)", str(url))
    if not match:
        return None
    parts = [p for p in match.group(1).split("-") if p]
    # Trailing hash-like fragments are LinkedIn's, not the person's.
    while parts and any(ch.isdigit() for ch in parts[-1]):
        parts.pop()
    if len(parts) < 2:
        return None
    return " ".join(p.capitalize() for p in parts)


def read_headline_record(profile: dict) -> tuple:
    """Pull a name, job title and company out of a search-result headline.

    ColdIQ's `provider: "auto"` waterfall can answer a people search from a live
    LinkedIn profile search, which returns snippets rather than person records:

        {"title": "Seb Hall - Founder @ Cloud Employee | Helping US & UK ...",
         "linkedin_url": "https://www.linkedin.com/in/seb-hall"}

    Two fields, and the name is glued to the front of `title`. Read literally
    that produced a lead with no name, no company, and the whole headline as the
    job title — which the frontend then dropped for having neither name nor
    company, so a search the logs reported as "coldiq returned 6 leads" showed
    the user nothing.

    Returns (name, job_title, company), any of which may be None. Nothing is
    guessed: the company is only taken from an explicit "@"/"at" marker or a
    tail that names a corporate form.
    """
    p = profile or {}
    raw = str(p.get("title") or p.get("headline") or "").strip()
    linked_name = name_from_linkedin_url(p.get("linkedin_url") or p.get("linkedinUrl"))

    if not raw:
        return linked_name, None, None

    # "<Name> - <headline>". Only the first separator splits: a headline can
    # contain more of them ("Founder - building X - hiring").
    name, sep, headline = raw.partition(" - ")
    if not sep:
        # No separator: the whole string is a headline, or it is just a title.
        name, headline = None, raw
    name = (name or "").strip() or None

    # A "name" of four or more words is a headline that happened to contain a
    # dash, not somebody's name.
    if name and len(name.split()) > 3:
        name, headline = None, raw

    # Everything after a pipe is positioning copy, not a role.
    headline = headline.split("|")[0].strip()

    company = None
    for marker in (" @ ", " at "):
        if marker in headline:
            headline, _, company = headline.partition(marker)
            headline, company = headline.strip(), company.strip()
            break
    else:
        # "Owner, WitWerx, Inc." — only when the tail names a corporate form.
        head, comma, tail = headline.partition(", ")
        if comma and _COMPANY_SUFFIX.search(tail.strip()):
            headline, company = head.strip(), tail.strip()
        elif comma:
            # A comma with nothing company-shaped after it still ends the title.
            headline = head.strip()

    return (name or linked_name), (headline or None), (company or None)


def transform_coldiq_profile(profile: dict, search_params: dict = None) -> dict:
    """Map one ColdIQ record onto our lead shape.

    Read tolerantly on purpose. ColdIQ's OpenAPI declares the search response as
    `data: nullable` with the description "Normalized, provider-agnostic
    result" — the per-person field names are not in the spec, and the API is
    unreachable from CI, so the exact keys cannot be confirmed here. Every
    spelling ColdIQ uses for the same idea elsewhere in its own schemas
    (first_name/last_name, full_name, linkedin_url, company_name, domain) is
    accepted, and `coldiq_person_search` logs the keys of the first record it
    ever sees so the real shape is one production search away.

    That log is what turned up the case this now handles: the waterfall can
    answer from a live LinkedIn profile search, whose records are only
    {title, linkedin_url} with the name glued to the front of the title. Those
    fields are read first where they exist, and read_headline_record fills the
    gaps only where they do not — a proper person record is unaffected.
    """
    p = profile or {}
    first = p.get("first_name") or p.get("firstName") or ""
    last = p.get("last_name") or p.get("lastName") or ""
    name = (p.get("full_name") or p.get("fullName") or p.get("name")
            or f"{first} {last}".strip())
    company = (p.get("company_name") or p.get("companyName") or p.get("company")
               or p.get("organization") or "")
    if isinstance(company, dict):
        company = company.get("name") or company.get("company_name") or ""
    job_title = (p.get("job_title") or p.get("jobTitle") or p.get("position")
                 or "")
    phone, phone_type = pick_phone(p)

    # Only for the thin shape: every structured field above wins outright.
    if not (name and company and job_title):
        headline_name, headline_title, headline_company = read_headline_record(p)
        name = name or headline_name or ""
        job_title = job_title or headline_title or ""
        company = company or headline_company or ""
    # `title` is the raw headline on a thin record and a real job title on a
    # full one, so it is only trusted after the headline reader has had a look.
    job_title = job_title or p.get("title") or ""

    return {
        "contact_name": name or None,
        "first_name": first or None,
        "last_name": last or None,
        "job_title": job_title or None,
        "company_name": company or None,
        "company_domain": domain_host(
            p.get("company_domain") or p.get("domain") or p.get("website") or ""
        ) or None,
        "business_email": p.get("email") or p.get("work_email") or p.get("business_email") or None,
        "phone": phone,
        "phone_type": phone_type,
        "phone_available": bool(phone),
        "linkedin_url": p.get("linkedin_url") or p.get("linkedinUrl")
                        or p.get("profile_url") or p.get("linkedin") or None,
        "industry": p.get("industry") or p.get("company_industry") or None,
        "country": p.get("country") or p.get("location") or None,
        "provider": "coldiq",
        # Which stated ICP dimensions ColdIQ could only rank by, not filter on.
        # Empty for a search it matched exactly.
        "unverified_dimensions": profile.get("_unverified_dimensions") or [],
    }


async def coldiq_person_search(params: dict, limit: int) -> dict:
    """Call ColdIQ POST /v1/people/search and return raw records.

    `fields: "compact"` is the documented default and the shape our transform
    reads. `provider: "auto"` uses ColdIQ's own managed waterfall — pinning a
    single vendor would make this leg a second, worse copy of a provider we
    already call directly.
    """
    if coldiq_out_of_credits():
        raise HTTPException(
            status_code=402,
            detail="ColdIQ credits exhausted — top up the ColdIQ account to keep searching")

    search_input = {**build_coldiq_filters(params), "limit": max(min(limit, 100), 1)}
    # Internal marker, not an API field: which ICP dimensions went in as a
    # keyword hint rather than a hard filter. Popped before the request and
    # carried onto the returned rows so a caller can see they are
    # segment-probable rather than segment-exact.
    unverified = search_input.pop("_unverified_dimensions", [])

    body = {
        "input": search_input,
        "provider": "auto",
        "fields": "compact",
    }
    headers = {
        "Authorization": f"Bearer {settings.coldiq_api_key or ''}",
        "Content-Type": "application/json",
    }
    print(f"ColdIQ people search body: {json.dumps(body)[:400]}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{COLDIQ_BASE}/v1/people/search", headers=headers, json=body,
        )
        print(f"ColdIQ search status: {resp.status_code} {resp.text[:300]}")
        coldiq_note_status(resp.status_code)
        if resp.status_code == 401:
            raise HTTPException(status_code=502, detail="ColdIQ rejected the API key")
        if resp.status_code == 402:
            raise HTTPException(
                status_code=402,
                detail="ColdIQ credits exhausted — top up the ColdIQ account to keep searching",
            )
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=resp.status_code if resp.status_code < 500 else 502,
                detail=f"ColdIQ search error: {resp.text[:300]}",
            )
        data = resp.json()

    # `data` for a single call, `results` for a bulk one. Accept a bare list too:
    # the spec types neither, so the shape is confirmed by what arrives.
    payload = data.get("data") if isinstance(data, dict) else data
    if payload is None and isinstance(data, dict):
        payload = data.get("results")
    if isinstance(payload, dict):
        payload = (payload.get("people") or payload.get("contacts")
                   or payload.get("results") or payload.get("data") or [])
    profiles = [x for x in (payload or []) if isinstance(x, dict)]
    if unverified:
        for record in profiles:
            record["_unverified_dimensions"] = unverified

    # Shape only, never values — this is how the documented-but-untyped record
    # gets confirmed from a real response instead of from a guess.
    if profiles:
        print(f"ColdIQ record keys: {sorted(profiles[0].keys())}")
    else:
        print("ColdIQ returned no usable records; "
              f"top-level keys were {sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}")

    meta = data.get("_meta") if isinstance(data, dict) else None
    if meta:
        print(f"ColdIQ _meta: {json.dumps(meta)[:200]}")

    return {"profiles": profiles, "total": len(profiles), "next_cursor": None}


async def fetch_from_coldiq(params: dict) -> list:
    """People-search fetcher for /search — returns raw ColdIQ records.

    Records that cannot become a lead are dropped here rather than counted.
    A lead needs a name or a company: the frontend shows nothing for a row with
    neither, so counting them made the log say "coldiq returned 6 leads" for a
    search the user saw as empty. The count and the screen have to agree, or the
    logs point away from the bug instead of at it.
    """
    limit = max(min(params.get("limit", 10), 100), 1)
    data = await coldiq_person_search(params, limit)

    usable, dropped = [], 0
    for record in data["profiles"]:
        lead = transform_coldiq_profile(record, params)
        if lead.get("contact_name") or lead.get("company_name"):
            usable.append(record)
        else:
            dropped += 1
    if dropped:
        print(f"ColdIQ dropped {dropped} record(s) with no name or company")
    return usable


async def coldiq_reveal(request: "EnrichRequest") -> dict:
    """Reveal one ColdIQ lead: find the email, enrich the profile if that misses.

    /v1/email/find is tried first because it is "charged only on a found, valid
    email" — a miss costs nothing, where /v1/person/enrich is charged whether or
    not it turns up contact details. On a miss we fall back to the enrich call,
    which still returns title/company/location and sometimes an email the finder
    would not commit to.

    Returns our lead shape, or {} when neither call produced anything.
    """
    identity: dict = {}
    if request.linkedin_url:
        identity["linkedin_url"] = request.linkedin_url
    if request.full_name:
        parts = request.full_name.split(" ")
        identity["first_name"] = parts[0]
        if len(parts) > 1:
            identity["last_name"] = " ".join(parts[1:])
    if request.company_domain:
        identity["domain"] = domain_host(request.company_domain)
    elif request.company:
        identity["company_name"] = request.company

    # PersonIdentity needs a LinkedIn URL, or a name paired with a company.
    findable = bool(identity.get("linkedin_url") or (
        identity.get("first_name") and
        (identity.get("domain") or identity.get("company_name"))))

    record: dict = {}
    if findable:
        found = await coldiq_verb("/v1/email/find", identity)
        if isinstance(found, dict):
            record = found

    if not record.get("email") and (identity or request.email):
        enrich_identity = dict(identity)
        if request.email:
            enrich_identity["email"] = request.email
        enriched = await coldiq_verb("/v1/person/enrich", enrich_identity)
        if isinstance(enriched, dict):
            # The finder's email wins if it had one; otherwise take everything.
            record = {**enriched, **{k: v for k, v in record.items() if v}}

    if not record:
        return {}
    return transform_coldiq_profile(record)


# ColdIQ answers 402 for every call once the balance runs out, and it answers it
# slowly: three doomed round trips per enrich and one per search, 1-3 seconds
# each, on every request the product serves. The balance is a fact the 402
# itself reports, so there is no reason to keep asking.
#
# Latched rather than permanent: a top-up should bring the provider back without
# a deploy, so the latch simply expires and the next call finds out.
COLDIQ_EXHAUSTED_SECONDS = 900.0
_coldiq_exhausted_until = 0.0


def coldiq_note_status(status: int) -> None:
    """Record a 402 so the next few calls can skip the round trip."""
    global _coldiq_exhausted_until
    if status == 402:
        _coldiq_exhausted_until = time.monotonic() + COLDIQ_EXHAUSTED_SECONDS
        print("ColdIQ is out of credits — skipping it for "
              f"{int(COLDIQ_EXHAUSTED_SECONDS // 60)} minutes")


def coldiq_out_of_credits() -> bool:
    """True while a recent 402 says there is nothing to spend."""
    return time.monotonic() < _coldiq_exhausted_until


async def coldiq_verb(path: str, identity: dict) -> Optional[dict]:
    """POST one record to a ColdIQ GTM verb and return its normalized `data`.

    Every verb shares the same envelope — {input, provider, ...} in, a
    VerbResponse out — so one caller covers find/enrich/verify. Returns None on
    anything that is not a usable result, including 404 ("No usable result found
    across providers"), which is a legitimate miss rather than an error.
    """
    if coldiq_out_of_credits():
        return None

    body = {"input": identity, "provider": "auto"}
    headers = {
        "Authorization": f"Bearer {settings.coldiq_api_key or ''}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{COLDIQ_BASE}{path}", headers=headers, json=body)
    except httpx.HTTPError as exc:
        print(f"ColdIQ {path} unreachable: {exc}")
        return None

    print(f"ColdIQ {path} status: {resp.status_code} {resp.text[:300]}")
    coldiq_note_status(resp.status_code)
    if resp.status_code == 404:
        return None
    if resp.status_code not in (200, 201):
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None

    data = payload.get("data") if isinstance(payload, dict) else payload
    if data is None and isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and results:
            data = results[0]
    if isinstance(data, dict):
        meta = payload.get("_meta") if isinstance(payload, dict) else None
        if isinstance(meta, dict):
            data.setdefault("_meta", meta)
        return data
    return None


# =============================================================================
# Email verification  (ColdIQ, over every provider's leads)
# =============================================================================
#
# Every provider we call sells contact data, and none of them agree on what a
# usable email is: Bytemine and Crustdata hand back whatever their index holds,
# Wiza grades its own reveals, and ColdIQ's search inherits whichever vendor its
# waterfall picked. Sending on that mix is how a domain's reputation gets spent.
#
# So one check sits in front of all four: whatever the reveal produced goes
# through ColdIQ /v1/email/verify before it is returned, and the verdict rides on
# the lead. The check is advisory, never destructive — an address that comes back
# undeliverable is still returned, flagged, because the caller may already have
# it and needs to be told it is bad rather than told nothing.

# ColdIQ's documented verification states (from the bulk-results `state` filter):
# deliverable, risky, undeliverable, unknown. Vendors behind the waterfall spell
# the same verdicts a dozen ways, so their spellings are mapped onto those four
# plus catch_all, which ColdIQ documents as conclusive and sendable.
_CIQ_VERDICT = {
    "deliverable": "deliverable", "valid": "deliverable", "verified": "deliverable",
    "ok": "deliverable", "safe": "deliverable", "good": "deliverable",
    "undeliverable": "undeliverable", "invalid": "undeliverable",
    "bad": "undeliverable", "bounced": "undeliverable", "not_found": "undeliverable",
    "catch_all": "catch_all", "catch-all": "catch_all", "catchall": "catch_all",
    "accept_all": "catch_all", "accept-all": "catch_all",
    "risky": "risky", "disposable": "risky", "role": "risky", "role_based": "risky",
    "spam_trap": "risky", "unverifiable": "risky",
    "unknown": "unknown", "pending": "unknown", "none": "unknown",
}

# What may be sent. catch_all and risky are sendable on purpose: ColdIQ's own
# bulk docs call them conclusive, and refusing them would drop most addresses at
# companies that run a catch-all server, which is a large share of B2B.
_CIQ_SENDABLE = {"deliverable": True, "catch_all": True, "risky": True,
                 "undeliverable": False, "unknown": None}


def read_coldiq_verdict(data: dict) -> dict:
    """Normalize one /v1/email/verify result into our verification shape.

    Untyped in the spec (`data: nullable`), so this reads every spelling ColdIQ's
    own schemas use for the idea — status/state/result/email_status — and treats
    an unrecognized one as `unknown` rather than as a pass or a fail.
    """
    d = data or {}
    raw = (d.get("status") or d.get("state") or d.get("result")
           or d.get("email_status") or d.get("verification_status")
           or d.get("deliverability") or "")
    if isinstance(raw, bool):
        raw = "deliverable" if raw else "undeliverable"
    status = _CIQ_VERDICT.get(str(raw).strip().lower().replace(" ", "_"), "unknown")

    # A verdict-less result that still says the address is a catch-all or a role
    # account is not "unknown" — those are the two flags worth carrying.
    if status == "unknown":
        if d.get("catch_all") or d.get("is_catch_all") or d.get("accept_all"):
            status = "catch_all"
        elif d.get("disposable") or d.get("is_disposable") or d.get("is_role"):
            status = "risky"

    meta = d.get("_meta") if isinstance(d.get("_meta"), dict) else {}
    score = d.get("score") or d.get("confidence")
    return {
        "status": status,
        "sendable": _CIQ_SENDABLE.get(status),
        "checked_by": "coldiq",
        "vendor": meta.get("provider") or d.get("provider"),
        "score": score if isinstance(score, (int, float)) else None,
        "raw_status": str(raw) or None,
    }


async def coldiq_verify_email(email: str) -> dict:
    """Verify one address. Never raises: a failed check is `unknown`, not an error.

    Verification is a quality gate on a reveal that has already been paid for. If
    ColdIQ is down, out of credits or unconfigured, the right outcome is the lead
    with an honest "we could not check this" on it — not a 502 that loses the
    reveal the user was charged for.
    """
    if not email or "@" not in email:
        return {"status": "unverified", "sendable": None, "checked_by": None,
                "reason": "no email to check"}
    if not provider_configured("coldiq"):
        return {"status": "unverified", "sendable": None, "checked_by": None,
                "reason": "coldiq not configured"}

    data = await coldiq_verb("/v1/email/verify", {"email": email})
    if data is None:
        return {"status": "unknown", "sendable": None, "checked_by": "coldiq",
                "reason": "coldiq returned no verdict"}
    verdict = read_coldiq_verdict(data)
    verdict["checked_at"] = datetime.now(timezone.utc).isoformat()
    return verdict


async def verify_revealed_lead(lead: dict, provider: str = None) -> dict:
    """Attach a deliverability verdict to a revealed lead, whoever sourced it.

    Mutates and returns the lead. `email_verified` is the one field a caller has
    to read: True to send, False to hold, None when nobody could say.

    `provider` names the leg that revealed this lead, for the log. It is passed
    in rather than read off the lead because only ColdIQ's transform sets a
    provider key — the others left the line reading "verify None lead", which
    told you a verdict happened but not what it was about.
    """
    email = (lead or {}).get("business_email") or (lead or {}).get("email")
    verdict = await coldiq_verify_email(email or "")

    # ColdIQ is the primary checker, but it is also the thing that runs out of
    # credits — production spent weeks answering "unknown" for every address
    # because of it. Findymail answers the same question and bills separately,
    # so a verdict ColdIQ could not reach is worth one more attempt rather than
    # being reported as unknown.
    # Fiber is asked before Findymail because it answers the same question with
    # ColdIQ's detail — catch-all, role-based, disposable — where Findymail
    # returns a bare boolean. Both are only reached when ColdIQ could not
    # answer; a verdict ColdIQ actually reached is never second-guessed.
    for name, check in (("fiber", fiber_verify_email),
                        ("findymail", findymail_verify_email)):
        if not (email and verdict.get("sendable") is None
                and verdict.get("status") in ("unknown", "unverified")):
            break
        if not provider_configured(name):
            continue
        fallback = await check(email)
        if fallback.get("sendable") is not None:
            fallback["checked_at"] = datetime.now(timezone.utc).isoformat()
            fallback["fell_back_from"] = verdict.get("reason") or verdict.get("status")
            verdict = fallback

    lead["email_verification"] = verdict
    lead["email_verified"] = verdict.get("sendable")
    if email:
        source = provider or lead.get("provider") or "unknown provider"
        print(f"{verdict.get('checked_by') or 'no'} verify {source} lead: "
              f"{verdict.get('status')} (sendable={verdict.get('sendable')})")
    return lead


# Our industries are lowercase PDL/LinkedIn values; Bytemine wants LinkedIn's
# own Title Case spelling. Only the mappings the docs actually evidence are
# listed — an industry that is not here is refused rather than guessed at, so a
# wrong casing can never silently zero a search again.
_BM_INDUSTRY = {
    "information technology and services": "Information Technology and Services",
    "computer software": "Computer Software",
    "internet": "Internet",
    "financial services": "Financial Services",
    "banking": "Banking",
    "hospital & health care": "Hospital & Health Care",
    "health care": "Health Care",
    "biotechnology": "Biotechnology",
    "pharmaceuticals": "Pharmaceuticals",
    "manufacturing": "Manufacturing",
    "retail": "Retail",
    "construction": "Construction",
    "real estate": "Real Estate",
    "education management": "Education",
    "e-learning": "E-Learning",
    "marketing and advertising": "Marketing and Advertising",
    "management consulting": "Management Consulting",
    "insurance": "Insurance",
    "automotive": "Automotive",
    "telecommunications": "Telecommunications",
    "food & beverages": "Food & Beverages",
    "hospitality": "Hospitality",
    "legal services": "Legal Services",
    "logistics and supply chain": "Logistics and Supply Chain",
    "computer & network security": "Computer & Network Security",
    "computer games": "Computer Games",
    "media production": "Media Production",
    "oil & energy": "Oil & Energy",
    "staffing and recruiting": "Staffing and Recruiting",
    "non-profit organization management": "Non-Profit Organization Management",
    "government administration": "Government Administration",
}


def linkedin_industry(value: str) -> Optional[str]:
    """The LinkedIn spelling of an industry, or None when we cannot map it.

    The table above is LinkedIn's industry vocabulary, not something private to
    one provider: Bytemine names industries this way, and so does Crustdata's
    company_industries — the field transform_crustdata_profile reads straight
    back out as a display value. We hold industries lowercase PDL-style, so
    anything crossing into either provider has to be translated first.
    """
    key = str(value or "").strip().lower()
    if not key:
        return None
    if key in _BM_INDUSTRY:
        return _BM_INDUSTRY[key]
    # Already Title Case and one of theirs.
    if value in _BM_INDUSTRY.values():
        return value
    return None


def bytemine_industry(value: str) -> Optional[str]:
    """Bytemine's spelling of an industry, or None when we cannot map it."""
    return linkedin_industry(value)


def build_bytemine_filters(p: dict) -> dict:
    """Translate internal search params into a Bytemine /contacts/search body.

    Raises ProviderUnsupported for anything Bytemine's contact search cannot
    express — notably country-level location and free-text keywords, which it
    has no field for. Crustdata handles both, so those searches fall through.
    """
    body: dict = {}

    if p.get("job_title"):
        body["jobTitles"] = [p["job_title"]]
    # Skipped when the job title already implies it — see
    # seniority_implied_by_title for why the redundant AND zeroed the search.
    if p.get("seniority") and not seniority_implied_by_title(
            p.get("job_title"), p["seniority"]):
        mapped = _BM_SENIORITY.get(str(p["seniority"]).strip().lower())
        if mapped:
            body["seniorityLevels"] = [mapped]
    if p.get("departments"):
        depts = p["departments"]
        body["departments"] = depts if isinstance(depts, list) else [depts]
    # Bytemine names industries the way LinkedIn does — "Information Technology
    # and Services", "Financial Services". We hold them lowercase PDL-style, and
    # sending those through unmapped matched nothing: every contact search in
    # production came back totalCount 0, for every ICP, since the provider was
    # added. A filter that silently zeroes the search is worse than one that
    # refuses, because the chain reads it as "this provider has no such people".
    if p.get("industry"):
        mapped = bytemine_industry(p["industry"])
        if not mapped:
            raise ProviderUnsupported("industry", p["industry"])
        body["industries"] = [mapped]
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

    # /contacts/search filters location by US state or city only — there is no
    # country field. Anything that is not one of those two is refused so the
    # chain reaches a provider that can express it.
    #
    # The old rule sent any token longer than two characters as a city, so
    # "Germany" became cities:["Germany"] and "Texas" a city too: a filter that
    # matches nothing, silently, while reporting a successful search.
    location = p.get("location") or p.get("company_location")
    if location:
        kind, value = classify_location(str(location))
        if kind == "state":
            body["states"] = [value]
        elif kind == "city":
            body["cities"] = [value]
        else:
            # A country, or a code that could be either. Neither is expressible.
            raise ProviderUnsupported("location", location)

    # /contacts/search has no free-text field. Keywords are meant to be resolved
    # to company domains first — bytemine_resolve_keywords does that and clears
    # the key. Reaching here with one still set means an unresolved term would
    # be silently dropped, so refuse rather than run a broader search.
    if p.get("keywords"):
        raise ProviderUnsupported("keywords", p["keywords"])

    # The user's sentence is not sent to Bytemine and not refused either. It is
    # not resolvable here — a segment term like "AI SaaS" can be matched against
    # company descriptions, but a whole request matches nothing that way and
    # would burn a company-search credit finding out — but it is also not an
    # extra criterion being dropped: the structured filters it was parsed into
    # are in this body already. Refusing made Bytemine sit out every search from
    # the UI, which always sends a query, so the chain quietly ran one provider
    # instead of three. It searches the filters and contributes what it finds.

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


def build_bytemine_company_body(params: dict) -> dict:
    """Build the /b2b-search body for a keyword-to-companies lookup.

    Separate from the call so the filters can be checked without spending a
    company-search credit to find out what was sent.
    """
    body: dict = {
        "keywords": (params.get("keywords") or "").strip(),
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

    # /b2b-search does have a country field, so unlike /contacts/search it can
    # narrow the company list by one — but a code that names both a state and a
    # country is no more resolvable here, and picking wrong would seed the
    # contact search with companies on the wrong continent.
    #
    # A region reaches the else and is refused. It used to reach the city
    # branch, which is how city:"Europe" was sent on every European search —
    # total_companies 0 every time, reported as a successful search with no
    # matches rather than as a filter this endpoint cannot express.
    location = params.get("company_location") or params.get("location")
    if location:
        kind, value = classify_location(str(location))
        if kind == "state":
            body["state"] = value
        elif kind == "country":
            body["country"] = value
        elif kind == "city":
            body["city"] = value
        else:
            raise ProviderUnsupported("location", location)
    return body


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

    body = build_bytemine_company_body(params)
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
    # /contacts/search answers {"contacts": [...], "totalCount": N, "page": ...}.
    # This read `data["data"]` and `data["pagination"]["total"]`, neither of
    # which this endpoint has ever returned — so `profiles` was [] and `total`
    # was 0 on every search since the provider was added, whatever the API
    # actually found. The probe settles it: jobTitles=["Founder"] returns
    # totalCount 1,297,933, and we reported nothing.
    #
    # Every previous theory about these zeros — the seniority double-filter, the
    # industry vocabulary, the missing country field — was about the request.
    # The request was fine. We were not reading the answer.
    profiles = (data.get("contacts") or data.get("results")
                or data.get("data") or [])
    pagination = data.get("pagination") or {}
    total = data.get("totalCount")
    if total is None:
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
        "phone_type": None,
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
    # Bytemine names the line type in the key: mobile_phone vs direct_dial.
    phone, phone_type = pick_phone(record)
    lead.update({
        "contact_name": record.get("full_name") or lead["contact_name"],
        "business_email": email if email and "*" not in str(email) else None,
        "email_status": "verified" if email and "*" not in str(email) else "no_email",
        "email_available": bool(email and "*" not in str(email)),
        "phone": phone,
        "phone_type": phone_type,
        "phone_available": bool(phone),
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

    # See seniority_implied_by_title: Crustdata's `=` on seniority_level is an
    # exact match against its own taxonomy, so a title-implied seniority ANDs
    # away every person the title matched.
    if p.get("seniority") and not seniority_implied_by_title(
            p.get("job_title"), p["seniority"]):
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
        # Crustdata cannot filter on industry, and refuses to say so.
        #
        # The shape probe asked it five ways, each ANDed with the same title
        # filter that returns 4,353,682 rows on its own:
        #
        #   company_industries                     (.)  "Computer Software"  -> 0
        #   company_industries                     (.)  "computer software"  -> 0
        #   company_industries                     =    "Computer Software"  -> 0
        #   company_professional_network_industry  (.)  "Computer Software"  -> 0
        #   company_professional_network_industry  =    "Computer Software"   -> 0
        #
        # Every one returns a 200 and zero rows. The field reads back fine in a
        # response — transform_crustdata_profile has always used it — but
        # returnable and filterable are not the same thing here, and an
        # unfilterable field silently annihilates the whole AND.
        #
        # So it is refused rather than sent. The chain then moves to a provider
        # that can express an industry instead of Crustdata reporting "no such
        # people" for an ICP it never actually searched. This is the same rule
        # build_bytemine_filters applies to a country it has no field for.
        raise ProviderUnsupported("industry", p["industry"])

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
    # Both carry free text. `keywords` is a filter the user stated; the
    # semantic query is the sentence the structured filters were parsed out of.
    # Crustdata takes one search string, so they are joined — with `mode:
    # "exact"` below, the structured filters stay hard constraints either way
    # and this only decides ranking within them.
    semantic_query = " ".join(
        part for part in (
            (params.get("keywords") or "").strip(),
            (params.get("semantic_query") or "").strip(),
        ) if part
    ).strip()
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
        "phone_type": None,
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
    phone, phone_type = pick_phone(contact)

    lead["business_email"] = primary.get("email") if primary else None
    lead["email_status"] = primary.get("status") if primary else None
    lead["email_available"] = bool(biz)
    lead["phone"] = phone
    lead["phone_type"] = phone_type
    lead["phone_available"] = bool(phone)
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
    """The LinkedIn URL on a raw provider record, whichever shape it arrives in.

    Crustdata nests it; ColdIQ puts it flat on the record. Reading only the
    nested path meant every ColdIQ result was untrackable, so campaign history
    never recorded one and the same people came back on every search.
    """
    nested = (((profile.get("social_handles") or {})
               .get("professional_network_identifier") or {}).get("profile_url"))
    return (nested or profile.get("linkedin_url") or profile.get("linkedinUrl")
            or profile.get("person_linkedin_url")
            or profile.get("contact_linkedin_url"))


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
    except IntegrityError:
        # Two identical searches raced: both missed the cache, both walked the
        # chain, and the slower one lost the insert. The frontend fires every
        # search twice from different connections, so this is the normal case
        # rather than an exceptional one. The row the winner wrote is the same
        # answer, so this is nothing to warn about — and the warning was noisy
        # enough to look like a real failure in the logs.
        print(f"Cache row for {search_hash[:12]}… already written by a "
              "concurrent identical search")
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


@app.middleware("http")
async def attach_treg_billing_context(request: Request, call_next):
    """Bind trusted tenant headers once; provider code never accepts model tags."""
    token = _treg_request_context.set({
        "customer_id": treg_customer_id_from_request(request),
        "workspace_id": request.headers.get("X-Workspace-ID"),
        "idempotency_key": request.headers.get("Idempotency-Key"),
    })
    try:
        return await call_next(request)
    finally:
        _treg_request_context.reset(token)


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
        "treg_configured": provider_configured("treg"),
        "degraded": degraded,
        "degraded_reason": (
            f"SEARCH_PROVIDER={preferred} but no API key is configured for it; "
            f"searches are served by {prov}"
        ) if degraded else None,
    }


class TregBudgetRequest(BaseModel):
    daily_cap_micro: Optional[int] = Field(default=None, ge=0)
    status: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value):
        if value not in (None, "active", "blocked"):
            raise ValueError("status must be active or blocked")
        return value


def require_billing_admin(provided: Optional[str]) -> None:
    if not settings.billing_admin_key:
        raise HTTPException(status_code=503, detail="Billing administration is not configured")
    if not provided or not hmac.compare_digest(provided, settings.billing_admin_key):
        raise HTTPException(status_code=401, detail="Invalid billing admin key")


def require_treg_billing_config() -> None:
    if not settings.treg_token or not settings.treg_org_id:
        raise HTTPException(status_code=503, detail="Treg billing is not configured")


@app.get("/billing/treg/customers/{customer_id}/usage")
async def treg_customer_usage(
    customer_id: str,
    days: int = 30,
    x_billing_admin_key: Optional[str] = Header(default=None),
):
    """Read invoiceable customer usage from Treg's ledger, never its call log."""
    require_billing_admin(x_billing_admin_key)
    require_treg_billing_config()
    customer_id = _valid_treg_meta_value("customer_id", customer_id, required=True)
    days = max(min(days, 366), 1)
    url = (
        f"{settings.treg_base_url.rstrip('/')}/orgs/{quote(settings.treg_org_id, safe='')}"
        f"/usage/by-tag"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            url,
            headers={"X-Treg-Token": settings.treg_token},
            params={"key": "customer", "days": days},
        )
    if response.is_error:
        raise _safe_treg_error(response)
    data = response.json()
    attributed = int(data.get("attributed_micro") or 0)
    unattributed = int(data.get("unattributed_micro") or 0)
    total = int(data.get("total_micro") or 0)
    if attributed + unattributed != total:
        raise HTTPException(status_code=502, detail="Treg usage ledger did not reconcile")
    row = next(
        (item for item in data.get("rows", []) if item.get("value") == customer_id),
        {"value": customer_id, "charged_micro": 0, "charged_usd": 0.0, "calls": 0},
    )
    return {
        "customer": row,
        "days": days,
        "ledger_reconciled": True,
        "unattributed_micro": unattributed,
        "unattributed_warning": unattributed != 0,
    }


@app.put("/billing/treg/customers/{customer_id}/budget")
async def set_treg_customer_budget(
    customer_id: str,
    budget: TregBudgetRequest,
    x_billing_admin_key: Optional[str] = Header(default=None),
):
    """Set Treg's advisory daily customer cap or block a customer."""
    require_billing_admin(x_billing_admin_key)
    require_treg_billing_config()
    customer_id = _valid_treg_meta_value("customer_id", customer_id, required=True)
    payload = budget.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail="Provide daily_cap_micro or status")
    url = (
        f"{settings.treg_base_url.rstrip('/')}/orgs/{quote(settings.treg_org_id, safe='')}"
        f"/budgets/customer/{quote(customer_id, safe='')}"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(
            url,
            headers={"X-Treg-Token": settings.treg_token},
            json=payload,
        )
    if response.is_error:
        raise _safe_treg_error(response)
    return response.json()


# =============================================================================
# ICP Parser Endpoint
# =============================================================================

# =============================================================================
# Enrich Endpoint  (Wiza Individual Reveal)
# =============================================================================

# Values a provider or an upstream fallback writes to mean "we don't know",
# which must never be read back as a fact. Anchored and case-insensitive: a real
# company called "NA Consulting" or a person named "Nan" has to survive.
_PLACEHOLDER_RE = re.compile(
    r"(?:unknown|n/?a|not\s*available|not\s*provided|none|null|nil|undefined|"
    r"-{1,3}|\?+|tbd|test|no\s*company|company|placeholder)",
    re.IGNORECASE,
)


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

    # A placeholder is not an identifier, and here it is worse than nothing: it
    # becomes a search filter. Production sent Wiza
    # {"full_name": "Ott Salmar", "company": "Unknown"} — a real person, scoped
    # to a company literally named Unknown, so the reveal could only miss.
    #
    # The string is written upstream because leads.company_name is NOT NULL and
    # the agent needs *something* when a provider returns no company. That is
    # reasonable for a column; it is not an identifier, and this is the one door
    # every reveal goes through, so it is the right place to drop it.
    @field_validator("company", "company_domain", "full_name", "email",
                     "linkedin_url", mode="after")
    @classmethod
    def _drop_placeholders(cls, value):
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned or _PLACEHOLDER_RE.fullmatch(cleaned):
            return None
        return cleaned


@app.post("/enrich")
async def enrich_lead(request: EnrichRequest):
    """
    Enrich a single lead, trying each configured provider in turn.

    Each provider reveals from the identifier its own search hands back, so the
    order follows what the lead is carrying rather than a fixed preference: a
    Bytemine PID unlocks the exact record that was shown, a LinkedIn URL goes to
    Crustdata, ColdIQ takes a LinkedIn URL or a name paired with a company, and
    anything left (email-only) falls through to the Wiza reveal, which accepts
    the widest set of identifiers.

    Whichever provider answers, the email it returns is verified through ColdIQ
    before this returns — see verify_revealed_lead. The verdict rides on the lead
    as `email_verification`; a bad address is flagged, never dropped.
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
            lead = await verify_revealed_lead(transform_bytemine_unlocked(record), "bytemine")
            return {
                "success": True,
                "provider": "bytemine",
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "email_verification": lead["email_verification"],
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
            lead = await verify_revealed_lead(transform_crustdata_enrich(person), "crustdata")
            return {
                "success": True,
                "provider": "crustdata",
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "email_verification": lead["email_verification"],
                "lead": lead,
            }
        # No Crustdata match — fall through with whatever identifiers we have.

    # Treg's routed enrichment fans out across its lead-gen catalog and returns
    # the provider it selected. It remains one leg in our outer waterfall.
    if "treg" in chain and (
        request.linkedin_url
        or request.email
        or (request.full_name and request.company_domain)
    ):
        payload = {
            "linkedin_url": request.linkedin_url,
            "email": request.email,
            "full_name": request.full_name,
            "domain": request.company_domain,
        }
        try:
            result = await treg_call(
                "treg.people.enrich",
                {k: v for k, v in payload.items() if v},
                "lead-enrich",
            )
            output = result.get("output") if isinstance(result, dict) else None
            person = output if isinstance(output, dict) else {}
        except HTTPException as exc:
            print(f"Treg reveal failed ({exc.status_code}); trying next provider")
            if not any(provider in chain for provider in ("coldiq", "wiza")):
                raise
            person = {}
        if person and any(person.values()):
            lead = transform_treg_person(person)
            lead = await verify_revealed_lead(lead, "treg")
            return {
                "success": True,
                "provider": "treg",
                "treg_served_by": (result.get("_treg") or {}).get("served_by"),
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "email_verification": lead["email_verification"],
                "lead": lead,
            }

    # ColdIQ reveal: its own managed waterfall behind one call, keyed on the
    # LinkedIn URL or the name + company its search hands back. Reached whether
    # or not the lead came from ColdIQ — a reveal is about the person, not about
    # which index found them.
    if "coldiq" in chain and (
        request.linkedin_url
        or request.email
        or (request.full_name and (request.company or request.company_domain))
    ):
        try:
            lead = await coldiq_reveal(request)
        except HTTPException as exc:
            print(f"ColdIQ reveal failed ({exc.status_code}); falling through to Wiza")
            lead = {}
        # An emailless ColdIQ hit is not worth ending the chain on while Wiza is
        # still there to try — the point of a reveal is the address.
        if lead and (lead.get("business_email")
                     or (lead.get("contact_name") and "wiza" not in chain)):
            lead = await verify_revealed_lead(lead, "coldiq")
            return {
                "success": True,
                "provider": "coldiq",
                "enrichment_status": "complete" if lead.get("business_email") else "no_email",
                "email_verification": lead["email_verification"],
                "lead": lead,
            }

    # Fiber reveals from a LinkedIn URL and grades the address while it is at
    # it: an email that comes back `valid` has already passed deliverability
    # verification upstream, which is the check ColdIQ cannot currently run.
    # Work emails only — this is B2B outbound, and a personal address costs more
    # and is worth less.
    # Phones are asked for even though they cost more than an email alone. This
    # leg returns early on a hit, so a reveal without them would quietly drop
    # the phone number Wiza would have supplied for the same lead — a cheaper
    # call that makes the product worse is not cheaper.
    if "fiber" in chain and request.linkedin_url:
        contact = await fiber_reveal(request.linkedin_url, want_phone=True)
        if contact.get("email"):
            lead = await verify_revealed_lead({
                "contact_name": contact.get("name") or request.full_name,
                "business_email": contact["email"],
                "company_name": request.company,
                "company_domain": request.company_domain,
                "phone": contact.get("phone"),
                "phone_type": classify_phone_type(contact.get("phone_type")),
                "linkedin_url": request.linkedin_url,
                "provider": "fiber",
            }, "fiber")
            return {
                "success": True,
                "provider": "fiber",
                "enrichment_status": "complete",
                "email_verification": lead["email_verification"],
                "lead": lead,
            }

    # Findymail last before Wiza, and worth reaching even when everything above
    # failed: it is charged only on a found email, so a miss here costs nothing.
    # It is also the natural partner to a ColdIQ lead, which can arrive as
    # nothing but a LinkedIn URL — that URL is exactly what
    # /api/search/business-profile takes.
    findymail_domain = findymail_domain_for(request.company_domain, request.company)
    if "findymail" in chain and (
        request.linkedin_url or (request.full_name and findymail_domain)
    ):
        contact = await findymail_find_email(
            linkedin_url=request.linkedin_url,
            name=request.full_name,
            domain=findymail_domain)
        if contact.get("email"):
            lead = await verify_revealed_lead({
                "contact_name": contact.get("name") or request.full_name,
                "business_email": contact.get("email"),
                "company_domain": domain_host(contact.get("domain") or "") or None,
                "company_name": request.company,
                "linkedin_url": request.linkedin_url,
                "provider": "findymail",
            }, "findymail")
            return {
                "success": True,
                "provider": "findymail",
                "enrichment_status": "complete",
                "email_verification": lead["email_verification"],
                "lead": lead,
            }

    if "wiza" not in chain:
        raise HTTPException(status_code=404, detail="No configured provider could enrich this lead")

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
                lead = await verify_revealed_lead(transform_reveal_contact(contact), "wiza")
                return {
                    "success": True,
                    "provider": "wiza",
                    "enrichment_status": "complete" if data.get("status") != "failed" else "failed",
                    "email_verification": lead["email_verification"],
                    "lead": lead,
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

    # The sentence is retained, always.
    #
    # It used to be popped and then only used when no structured field was set —
    # which is never, because the frontend parses the query into filters before
    # sending and ships both. So the words the user actually typed were dropped
    # on the floor, and every search with the same coarse filters became the
    # same search: identical params, identical cache key, identical rows for the
    # cache's whole hour. Three different sentences about three different
    # segments returned the same people because by this point they were the same
    # request.
    #
    # It stays as `semantic_query`, separate from `keywords`: keywords are a
    # filter the user stated, this is the original phrasing the structured
    # filters were derived from. Crustdata searches it; providers that cannot
    # step aside — see build_bytemine_filters.
    if raw_query and any(params.get(f) for f in STRUCTURED_FIELDS):
        params["semantic_query"] = raw_query
        print(f"Retaining semantic query alongside structured filters: {raw_query!r}")

        # Retaining the sentence only helps the one provider that reads it.
        # Crustdata searches semantic_query; every other leg sees the structured
        # filters and nothing else. So "Founders at B2B AI SaaS companies" went
        # out as job_title=Founder + industry=computer software, and when
        # Crustdata returned nothing — which is most searches — what the user
        # actually saw was any founder at any small software company anywhere.
        # The AI part of their ICP was never applied to a single lead they were
        # shown.
        #
        # The frontend does its own parsing and sends no keywords, so the
        # backend parser below (which keeps these terms — see the comment in
        # parse_icp) never runs. Recover them here.
        #
        # They are NOT promoted to `keywords`. That field means "a criterion the
        # user stated", and a provider with no free-text field refuses the whole
        # search rather than drop one — build_wiza_filters does exactly that. A
        # term lifted out of the sentence has not been stated separately; it has
        # the same provenance as semantic_query and gets the same treatment, so
        # it narrows the providers that can express it and sidelines nobody.
        if not params.get("keywords"):
            terms = semantic_terms_in(raw_query)
            if terms:
                params["semantic_keywords"] = " ".join(terms)
                print("Recovered segment terms the structured filters dropped: "
                      f"{params['semantic_keywords']!r}")

    # An industry nobody has a field for is a search term, not a filter.
    #
    # The frontend's parser invents segment names — "Beauty & Wellness", "Salon
    # & Spa", "Events" — that are in no provider's taxonomy. Each leg then
    # correctly refuses a filter it cannot express, and because they all refuse
    # the same one the user gets an empty page for every salon search: eight
    # providers, all of them right, nothing found.
    #
    # Refusing is the correct answer to "express this or step aside" and the
    # wrong answer to this. The term is not dropped — that would silently
    # broaden the ICP, which is what the refusal exists to prevent — it moves to
    # the free-text side, where GetLeads matches it against company descriptions
    # and Fiber against headlines and summaries. "Salon" is a good filter there;
    # it is only a bad enum value.
    if params.get("industry") and not industry_is_expressible(params["industry"]):
        demoted = str(params.pop("industry")).strip()
        existing = params.get("semantic_keywords")
        params["semantic_keywords"] = (
            f"{existing} {demoted}".strip() if existing else demoted)
        print(f"No provider has an industry field for {demoted!r} — searching it "
              f"as free text instead of refusing every leg")

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
        # The user's sentence narrows a title-only search through Crustdata's
        # semantic ranking, so it counts as a narrowing filter here.
        "semantic_query",
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
    if not chain:
        raise HTTPException(status_code=503, detail="No lead data provider is configured")
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
        """The transform that reads one provider's row shape.

        Bound to `params` so a merged result can transform each row with its own
        provider's reader — the rows are not interchangeable.
        """
        if name == "bytemine":
            return lambda profile: transform_bytemine_profile(profile, params)
        if name == "coldiq":
            return lambda profile: transform_coldiq_profile(profile, params)
        if name == "findymail":
            return lambda row: transform_findymail_row(row, params)
        if name == "fiber":
            return lambda row: transform_fiber_profile(row, params)
        if name == "getleads":
            return lambda record: transform_getleads_contact(record)
        if name == "treg":
            return lambda record: transform_treg_person(record, params)
        if name == "crustdata":
            return lambda profile: transform_crustdata_profile(profile, params)
        if degraded:
            return lambda profile: transform_preview_profile(profile)
        return lambda profile: transform_wiza_contact(profile, params)

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
        next_cursor = None
        if isinstance(cached_payload, dict) and cached_payload.get("buckets") is not None:
            # Merged shape: rebuild through each provider's own transform.
            data, leads, total, _ = merge_provider_results(
                cached_payload["buckets"], transform_for)
            total = cached_payload.get("total", total)
            next_cursor = cached_payload.get("next_cursor")
            served_by = cached_payload.get("provider") or prov
        elif isinstance(cached_payload, dict):
            # Written before the merge: one provider's rows, one transform.
            data = cached_payload.get("profiles") or []
            total = cached_payload.get("total", len(data))
            next_cursor = cached_payload.get("next_cursor")
            served_by = cached_payload.get("provider") or prov
            leads = [transform_for(served_by)(r) for r in data]
        else:
            # Written before cursor metadata was persisted.
            data = cached_payload
            total = len(data)
            leads = [transform_for(prov)(r) for r in data]
        return SearchResponse(
            success=True, source="cache", from_cache=True,
            count=len(leads), total=total, leads=leads, data=data,
            next_cursor=next_cursor, campaign_id=request.campaign_id,
            provider=served_by,
        )

    print(f"Cache MISS — walking chain {'+'.join(chain)}")

    # Each provider resumes from its own position — see decode_cursors.
    cursor_by_provider = decode_cursors(request.cursor, chain)

    # run_provider rebinds `params` to a per-leg copy carrying that leg's share
    # of the page, so it needs a stable name for the search's own params.
    outer_params = params

    async def run_provider(name: str, want: int = None):
        """One provider's search. Returns (raw_results, total, next_cursor).

        `want` is how many rows this leg is being asked for, which is the page
        size for the first leg and the shortfall for the ones after it. Later
        legs top up a partial page rather than re-requesting the whole thing:
        several of these bill per record returned, so asking for six when two
        are missing is four rows of waste on every partially-filled search.
        """
        provider_cursor = cursor_by_provider.get(name)
        if want is not None:
            params = {**outer_params, "limit": max(want, 1)}
        else:
            params = outer_params
        if name == "bytemine":
            result = await bytemine_person_search(
                params, max(min(params.get("limit", 10), 100), 1),
                cursor=provider_cursor, offset=request.start_offset)
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
            skip = 0 if provider_cursor else request.start_offset
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
                cursor=provider_cursor,
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

        if name == "getleads":
            # The one leg with a real offset, so it is the one leg that can go
            # and find people it has not shown before.
            #
            # It used to fetch offset=start_offset once and filter the result.
            # start_offset is zero on every fresh search — the frontend sends
            # page 1 each time — so the same first N rows came back for the same
            # ICP, the already-seen list removed the ones the user had already
            # been shown, and what was left was whatever remained of a fixed
            # page. Search the same ICP a few times and that page is entirely
            # seen: the provider is still answering, still billing, and the user
            # gets nothing new. That is the "same leads over and over" this has
            # been doing.
            #
            # So: page forward until the requested number of *unseen* people is
            # found. Bounded, because this provider bills per record returned
            # and each page is a round trip — a fully-seen page depth is a real
            # "nothing new here" answer, not a reason to walk the whole index.
            wanted = params.get("limit", 10)
            offset = request.start_offset
            seen = {i for i in (linkedin_identity(u) for u in (exclusions or [])) if i}
            found: list = []
            total = 0

            for _ in range(GETLEADS_MAX_PAGES):
                data = await getleads_person_search(params, wanted, offset=offset)
                page = data["profiles"]
                total = data["total"]

                fresh = page
                if seen:
                    fresh = [c for c in fresh
                             if linkedin_identity(_profile_url(c)) not in seen]
                if campaign_keys:
                    fresh = [c for c in fresh
                             if _profile_lead_key(c) not in campaign_keys]
                found.extend(fresh)

                next_offset = data.get("next_offset")
                if len(found) >= wanted or not page or next_offset is None:
                    break
                print(f"GetLeads: {len(page)} row(s) at offset {offset} already "
                      f"seen — paging to {next_offset} for new people")
                offset = next_offset

            found = found[:wanted]
            if request.campaign_id:
                found = await record_new_campaign_profiles(request.campaign_id, found)

            return found, total, None

        if name == "fiber":
            # A real cursor, so a numbered page resumes rather than being
            # over-fetched and sliced — and this leg bills per profile
            # returned, so an over-fetch here is money.
            #
            # A cursor only exists once the caller has paged, so on a first page
            # the already-seen filter is all that separates a repeat search from
            # the same rows again. That is the GetLeads defect in a different
            # shape; the difference is that a Fiber cursor, once we have one,
            # resumes exactly where the last page stopped.
            data = await fiber_person_search(
                params, params.get("limit", 10), cursor=provider_cursor)
            found = data["profiles"]

            if exclusions:
                seen = {i for i in (linkedin_identity(u) for u in exclusions) if i}
                found = [r for r in found
                         if linkedin_identity(_profile_url(r)) not in seen]
            if campaign_keys:
                found = [r for r in found
                         if _profile_lead_key(r) not in campaign_keys]
            if request.campaign_id:
                found = await record_new_campaign_profiles(request.campaign_id, found)

            return found, data["total"], data.get("next_cursor")

        if name == "treg":
            # Treg's routed endpoint chooses among its lead-gen providers. It
            # has no stable cross-provider cursor, so numbered pages over-fetch
            # and slice just like other non-cursor legs.
            skip = request.start_offset
            data = await treg_person_search(
                params, max(min(params.get("limit", 10) + skip, 100), 1),
                cursor=provider_cursor)
            found = data["profiles"]
            if exclusions:
                seen = {i for i in (linkedin_identity(u) for u in exclusions) if i}
                found = [p for p in found
                         if linkedin_identity(_profile_url(p)) not in seen]
            return (found[skip:] if skip else found), data["total"], data.get("next_cursor")

        if name == "findymail":
            # Intellimatch has no cursor and no offset, so a numbered page is
            # over-fetched and sliced. require_email is already on, so rows that
            # arrive have an address — the filtering below only removes people
            # already shown.
            skip = request.start_offset
            data = await findymail_person_search(
                params, max(min(params.get("limit", 10) + skip, 500), 1))
            found = data["profiles"]
            if exclusions:
                seen = {i for i in (linkedin_identity(u) for u in exclusions) if i}
                found = [r for r in found
                         if linkedin_identity(_profile_url(r)) not in seen]
            if campaign_keys:
                found = [r for r in found
                         if _profile_lead_key(r) not in campaign_keys]
            if request.campaign_id:
                found = await record_new_campaign_profiles(request.campaign_id, found)
            return (found[skip:] if skip else found), data["total"], None

        if name == "coldiq":
            # ColdIQ's people search has no cursor, no offset and no exclusion
            # field, so every "show me someone new" has to happen on this side.
            # Without it the leg returned the same top results on every search
            # — the same person appeared in five consecutive searches — and
            # once the other three providers came back empty, that was the
            # entire page the user saw, every time.
            #
            # Over-fetch to survive that filtering, but only to a multiple of
            # the page — not by the length of the seen list. Asking for one row
            # per previously-seen person meant a page of 6 requesting 56, and
            # this provider bills per record returned: the log showed a single
            # search costing 10 credits against a balance of 0.32.
            #
            # The trade is honest either way. A page that comes back fully
            # filtered is "nothing new here", which is true, and GetLeads —
            # which has a real offset and does not need this at all — is the
            # leg that should be finding new people now.
            skip = request.start_offset
            wanted = params.get("limit", 10) + skip
            data = await coldiq_person_search(
                params, max(min(wanted * COLDIQ_OVERFETCH, 100), 1))
            before = len(data["profiles"])
            found = data["profiles"]

            if exclusions:
                seen = {i for i in (linkedin_identity(u) for u in exclusions) if i}
                found = [p for p in found
                         if linkedin_identity(_profile_url(p)) not in seen]
            if campaign_keys:
                found = [p for p in found
                         if _profile_lead_key(p) not in campaign_keys]
            if request.campaign_id:
                found = await record_new_campaign_profiles(request.campaign_id, found)

            if before and not found:
                print(f"ColdIQ: all {before} record(s) already seen — "
                      "no new people at this page depth")

            return (found[skip:] if skip else found), data["total"], None

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

    attempts: list = []
    buckets: list = []
    refused: dict[str, str] = {}
    failures: dict[str, object] = {}
    ran = False

    # Waterfall: call one provider at a time, in configured order, and stop as
    # soon as the page is full. Move to the next leg when the current one cannot
    # express the ICP, errors, returns nothing — or returns less than was asked
    # for.
    #
    # That last case used to end the chain too: `if found: break` stopped on the
    # first leg with *any* row, so one lead from Bytemine ended a search for six
    # and the remaining five providers were never asked. The point of a chain is
    # that nobody depends on a single provider, and a page of one lead is that
    # dependency at its worst — the user sees a nearly empty result while four
    # tools that could have filled it sat idle.
    #
    # It stays a waterfall: nothing is called speculatively, each leg is asked
    # only for the shortfall, and the walk ends the moment the page is full.
    # merge_provider_results already round-robins the buckets and drops
    # cross-provider duplicates, so a topped-up page is one page, not two
    # stacked.
    wanted = max(int(params.get("limit") or 10), 1)
    collected = 0

    for name in chain:
        try:
            outcome = await run_provider(name, wanted - collected)
        except ProviderUnsupported as outcome:
            # This provider cannot express one of the requested filters. It sits
            # this search out; the others still answer it.
            print(f"{name} cannot express {outcome} — it contributes nothing here")
            refused[name] = outcome.field
            attempts.append({"provider": name, "outcome": "unsupported_filter",
                             "detail": outcome.field})
            continue
        except HTTPException as outcome:
            print(f"{name} failed ({outcome.status_code})")
            failures[name] = outcome
            attempts.append({"provider": name, "outcome": "error",
                             "detail": outcome.status_code})
            continue
        except Exception as outcome:
            print(f"{name} raised {type(outcome).__name__}: {outcome}")
            failures[name] = outcome
            attempts.append({"provider": name, "outcome": "error", "detail": "exception"})
            continue

        found, provider_total, provider_cursor = outcome
        ran = True
        attempts.append({"provider": name,
                         "outcome": "results" if found else "no_results",
                         "count": len(found)})
        if found or provider_cursor:
            buckets.append({"provider": name, "profiles": found,
                            "total": provider_total, "next_cursor": provider_cursor})
        collected += len(found)
        if collected >= wanted:
            break
        if found:
            print(f"{name} filled {collected}/{wanted} — continuing the chain "
                  f"for {wanted - collected} more")

    # Nothing ran at all. Distinguish the two ways that happens: every provider
    # refused the filter, or every provider errored. An empty success would
    # report "no such people" for both, which is true of neither.
    if not ran:
        if refused and not failures:
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
        if failures:
            # Surface a real upstream error rather than inventing an empty page.
            first = next(iter(failures.values()))
            if isinstance(first, HTTPException):
                raise first
            raise HTTPException(status_code=502, detail="Every lead provider failed for this search")

    raw_results, leads, total, cursor_map = merge_provider_results(buckets, transform_for)
    next_cursor = encode_cursors(cursor_map)
    contributors = [b["provider"] for b in buckets if b["profiles"]]
    served_by = "+".join(contributors) if contributors else (chain[0] if chain else "none")

    summary = ", ".join(
        f"{a['provider']}:{a.get('count', a['outcome'])}" for a in attempts
    )
    print(f"{served_by}{' (degraded preview)' if degraded and served_by == 'wiza' else ''} "
          f"returned {len(leads)} leads ({summary})")

    # Stored per provider, because a merged page can only be rebuilt by reading
    # each provider's rows with its own transform.
    cache_payload = {"buckets": buckets, "total": total,
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
# Provider filter diagnostics
# =============================================================================
#
# Bytemine and Crustdata return totalCount 0 for ICPs that GetLeads answers with
# a full page — the same job title, the same industry, the same headcount band.
# Three fixes have been aimed at that from a reading of the request bodies and
# none of them moved it, because a body that looks correct and a body that
# matches their index are not the same thing and no amount of re-reading tells
# them apart.
#
# So: stop guessing and ask. This adds one filter at a time and reports the
# count after each, which turns "the search returns nothing" into "the search
# returns nothing once <field> is added" — a fact rather than a hypothesis.
#
# It is safe to run. Every probe asks for a single row, and a probe that matches
# nothing returns nothing, which is the case being investigated and the case
# these providers do not bill for. A full sweep of a four-filter ICP across
# three providers costs at most twelve records.

DIAGNOSTIC_FILTER_ORDER = (
    "job_title", "seniority", "industry", "company_size",
    "location", "company_location", "departments", "keywords",
)

DIAGNOSTIC_PROVIDERS = ("bytemine", "crustdata", "getleads")


async def probe_provider_filters(name: str, params: dict) -> dict:
    """Run one provider's search for a single row and report what came back."""
    try:
        if name == "bytemine":
            result = await bytemine_person_search(params, 1)
        elif name == "crustdata":
            result = await crustdata_person_search(params, 1)
        elif name == "getleads":
            result = await getleads_person_search(params, 1)
        else:
            return {"outcome": "not_probed"}
    except ProviderUnsupported as refusal:
        return {"outcome": "unsupported_filter", "field": refusal.field}
    except HTTPException as failure:
        return {"outcome": "error", "status": failure.status_code,
                "detail": str(failure.detail)[:300]}
    except Exception as failure:  # noqa: BLE001 — a probe reports, never raises
        return {"outcome": "error", "detail": f"{type(failure).__name__}: {failure}"[:300]}

    return {"outcome": "ok", "total": result.get("total"),
            "returned": len(result.get("profiles") or [])}


@app.get("/diagnostics/provider-filters")
async def diagnose_provider_filters_from_url(
    query: str = None,
    job_title: str = None,
    industry: str = None,
    company_size: str = None,
    location: str = None,
    seniority: str = None,
    keywords: str = None,
):
    """The same probe, reachable from a browser address bar.

    The POST below is the real shape, but a diagnostic nobody can run does not
    diagnose anything: this is meant to be pasted into a URL bar by whoever is
    watching the logs, without a terminal or an HTTP client in the way.
    """
    return await diagnose_provider_filters(SearchRequest(
        query=query, job_title=job_title, industry=industry,
        company_size=company_size, location=location, seniority=seniority,
        keywords=keywords))


@app.post("/diagnostics/provider-filters")
async def diagnose_provider_filters(request: SearchRequest):
    """Find which filter empties a provider's search.

    Send the ICP that returns nothing. Each provider is searched with the first
    filter alone, then the first two, and so on; the step where the count drops
    to zero names the filter their index disagrees with us about.

    A provider whose count is zero on its very first filter has a problem with
    the whole integration rather than one field, which is just as useful to
    know and is not something the request body can tell us.
    """
    params = await resolve_search_params(request)
    present = [f for f in DIAGNOSTIC_FILTER_ORDER if params.get(f)]
    if not present:
        raise HTTPException(
            status_code=400,
            detail="Send the search that returns nothing — there are no filters here to isolate.")

    report: dict = {}
    for name in DIAGNOSTIC_PROVIDERS:
        if not provider_configured(name):
            report[name] = {"steps": [], "verdict": "not configured"}
            continue

        steps: list = []
        verdict = None
        applied: dict = {}
        for field in present:
            applied[field] = params[field]
            outcome = await probe_provider_filters(name, dict(applied))
            steps.append({"added": field, "value": applied[field], **outcome})
            print(f"probe {name}: +{field} -> {outcome}")

            if outcome["outcome"] == "ok" and not outcome.get("total"):
                verdict = (f"empty once {field}={applied[field]!r} is applied"
                           if len(applied) > 1 else
                           f"empty on {field}={applied[field]!r} alone — this is the "
                           "whole integration, not one filter")
                break
            if outcome["outcome"] != "ok":
                verdict = f"stopped at {field}: {outcome['outcome']}"
                break

        report[name] = {"steps": steps, "verdict": verdict or "no filter emptied it"}

    return {"params": {f: params[f] for f in present}, "providers": report}


# =============================================================================
# Provider request-shape probes
# =============================================================================
#
# The filter probe above narrowed the two zeros to different problems, and
# neither is a filter value:
#
#   bytemine  — empty on job_title alone. Nothing is wrong with the ICP; the
#               request itself is not reaching the index.
#   crustdata — 4,353,682 for the title, 0 the moment industry is added. The
#               "(.)" operator and the experience.employment_details.current.
#               prefix are therefore both fine, since the title filter uses
#               both. Only the industry field or its value is wrong.
#
# So this probe varies the *request shape* rather than the ICP, and reports the
# count each shape returns. The decisive one for Bytemine is the empty body: a
# filterless search that returns rows means the index is reachable and our
# filter keys are wrong, and one that returns nothing means the envelope, the
# path or the credentials are.
#
# Every probe asks for a single row.

BYTEMINE_SHAPE_PROBES = (
    # The decisive one: no filters at all. Answered 168,465,180 in production,
    # which is how we learned the request arrives and the response parsing was
    # the bug (see bytemine_person_search).
    ("no filters", {}),
    ("jobTitles (current)", {"jobTitles": ["Founder"]}),
    ("job_titles, the spelling GetLeads uses", {"job_titles": ["Founder"]}),
    ("titles", {"titles": ["Founder"]}),
    ("jobTitle, singular", {"jobTitle": "Founder"}),
    ("page 1 rather than 0", {"jobTitles": ["Founder"], "page": 1}),

    # employeeSizes is the next zero. With the response parsing fixed,
    # {"jobTitles": ["Owner"], "employeeSizes": ["1-10"]} still returns
    # totalCount 0 while the title alone returns 1,297,933 — so the band string
    # is the thing this endpoint does not recognise. These are the shapes worth
    # trying before guessing at a fix.
    ("employeeSizes 1-10 (current)",
     {"jobTitles": ["Founder"], "employeeSizes": ["1-10"]}),
    ("employeeSize, singular",
     {"jobTitles": ["Founder"], "employeeSize": "1-10"}),
    ("employeeSizes 1-10 spaced",
     {"jobTitles": ["Founder"], "employeeSizes": ["1 - 10"]}),
    ("employeeSizes 1_10",
     {"jobTitles": ["Founder"], "employeeSizes": ["1_10"]}),
    ("numeric min/max, the shape GetLeads takes",
     {"jobTitles": ["Founder"], "employeesMin": 1, "employeesMax": 10}),
    ("employeeCount min/max",
     {"jobTitles": ["Founder"], "employeeCountMin": 1, "employeeCountMax": 10}),
)

# Each is ANDed with the title filter that already works, so the only thing
# changing between rows is how the industry is expressed.
_CD_INDUSTRY_ALT = "experience.employment_details.current.company_professional_network_industry"

CRUSTDATA_INDUSTRY_PROBES = (
    ("company_industries (.) Title Case (current)", _CD_INDUSTRY, "(.)", "Computer Software"),
    ("company_industries (.) lowercase", _CD_INDUSTRY, "(.)", "computer software"),
    ("company_industries =", _CD_INDUSTRY, "=", "Computer Software"),
    ("company_professional_network_industry (.)", _CD_INDUSTRY_ALT, "(.)", "Computer Software"),
    ("company_professional_network_industry =", _CD_INDUSTRY_ALT, "=", "Computer Software"),
)


async def probe_bytemine_shape(body: dict) -> dict:
    """One Bytemine contact search with a hand-built body."""
    try:
        data = await bytemine_call("/contacts/search", {"pageSize": 1, "page": 0, **body})
    except HTTPException as failure:
        return {"outcome": "error", "status": failure.status_code,
                "detail": str(failure.detail)[:200]}
    except Exception as failure:  # noqa: BLE001 — a probe reports, never raises
        return {"outcome": "error", "detail": f"{type(failure).__name__}: {failure}"[:200]}
    return {"outcome": "ok", "total": data.get("totalCount"),
            "returned": len(data.get("contacts") or [])}


async def probe_crustdata_shape(filters: dict) -> dict:
    """One Crustdata person search with a hand-built filter tree."""
    body = {
        "limit": 1,
        "fields": ["crustdata_person_id", "basic_profile", "experience"],
        "filters": filters,
        "sorts": [{"field": "crustdata_person_id", "order": "asc"}],
    }
    headers = {
        "Authorization": f"Bearer {settings.crustdata_api_key}",
        "x-api-version": CRUSTDATA_VERSION,
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{CRUSTDATA_BASE}/person/search",
                                     headers=headers, json=body)
    except httpx.HTTPError as failure:
        return {"outcome": "error", "detail": f"unreachable: {failure}"[:200]}

    if resp.status_code != 200:
        return {"outcome": "error", "status": resp.status_code,
                "detail": resp.text[:200]}
    data = resp.json()
    return {"outcome": "ok", "total": data.get("total_count"),
            "returned": len(data.get("profiles") or [])}


@app.get("/diagnostics/provider-shapes")
async def diagnose_provider_shapes(job_title: str = "Founder"):
    """Vary the request shape, not the ICP, and report what each one finds.

    Reachable from a browser for the same reason the filter probe is: the
    person who needs the answer is the one reading the logs.
    """
    report: dict = {}

    if provider_configured("bytemine"):
        rows = []
        for label, body in BYTEMINE_SHAPE_PROBES:
            shaped = dict(body)
            for key in ("jobTitles", "job_titles", "titles"):
                if key in shaped:
                    shaped[key] = [job_title]
            if "jobTitle" in shaped:
                shaped["jobTitle"] = job_title
            outcome = await probe_bytemine_shape(shaped)
            print(f"shape bytemine [{label}] -> {outcome}")
            rows.append({"shape": label, "body": shaped, **outcome})

        empty = next((r for r in rows if r["shape"] == "no filters"), {})
        if empty.get("outcome") == "ok" and empty.get("total"):
            verdict = ("the index is reachable and a filterless search finds "
                       "people — so the filter keys are what it is not reading")
        elif empty.get("outcome") == "ok":
            verdict = ("even a filterless search finds nothing — this is the "
                       "envelope, the path or the credentials, not the filters")
        else:
            verdict = f"could not complete the filterless probe: {empty.get('detail')}"

        sizes = [r["shape"] for r in rows
                 if "employee" in r["shape"].lower()
                 and r.get("outcome") == "ok" and r.get("total")]
        verdict += ("; headcount expressible as: " + ", ".join(sizes) if sizes
                    else "; no headcount shape returns anything")
        report["bytemine"] = {"probes": rows, "verdict": verdict}

    if provider_configured("crustdata"):
        rows = []
        title = {"field": _CD_TITLE, "type": "(.)", "value": job_title}
        for label, field, op, value in CRUSTDATA_INDUSTRY_PROBES:
            filters = {"op": "and", "conditions": [
                title, {"field": field, "type": op, "value": value}]}
            outcome = await probe_crustdata_shape(filters)
            print(f"shape crustdata [{label}] -> {outcome}")
            rows.append({"shape": label, "field": field, "operator": op,
                         "value": value, **outcome})

        works = [r["shape"] for r in rows
                 if r.get("outcome") == "ok" and r.get("total")]
        report["crustdata"] = {
            "probes": rows,
            "verdict": (f"these express the industry: {', '.join(works)}" if works
                        else "no spelling of the industry filter returns anything"),
        }

    return {"job_title": job_title, "providers": report}


# =============================================================================
# Run with: uvicorn main:app --reload
# =============================================================================
