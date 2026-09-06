import json
import os
import unittest
import base64
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/test")
os.environ.setdefault("WIZA_API_KEY", "test-wiza")
os.environ.setdefault("CRUSTDATA_API_KEY", "test-crustdata")

import main
from fastapi import HTTPException


def flatten_conditions(filters):
    if not filters:
        return []
    return filters.get("conditions", [filters])


class CrustdataFilterTests(unittest.TestCase):
    def test_correct_fields_and_values_are_used(self):
        filters = main.build_crustdata_filters({
            "job_title": "Founder",
            "seniority": "vp",
            "location": "US",
            "company_location": "New York",
            "industry": "computer software",
            "company_size": "1-10",
            "keywords": "outbound automation",
        })
        conditions = flatten_conditions(filters)
        by_field = {condition["field"]: condition for condition in conditions}

        self.assertEqual(by_field["basic_profile.location.country"]["value"], "United States")
        # company_industries is LinkedIn's vocabulary — "Computer Software".
        # This used to assert the raw lowercase term went through untranslated,
        # which is a contains match on a string the field never holds.
        self.assertEqual(
            by_field["experience.employment_details.current.company_industries"]["value"],
            "Computer Software",
        )
        self.assertEqual(
            by_field["experience.employment_details.current.seniority_level"]["value"],
            "VP",
        )
        self.assertIn(
            "experience.employment_details.current.company_hq_location", by_field
        )
        title_values = [
            condition["value"] for condition in conditions
            if condition["field"] == "experience.employment_details.current.title"
        ]
        self.assertEqual(title_values, ["Founder"])

    def test_non_country_location_uses_full_location(self):
        condition = main.build_crustdata_filters({"location": "San Francisco"})
        self.assertEqual(condition["field"], "basic_profile.location.full_location")

    def test_title_only_search_requires_explicit_opt_in(self):
        with self.assertRaises(main.HTTPException) as raised:
            main.enforce_crustdata_search_scope({"job_title": "Founder"}, False)
        self.assertEqual(raised.exception.status_code, 422)

        main.enforce_crustdata_search_scope({"job_title": "Founder"}, True)

    def test_ignored_filters_are_rejected(self):
        with self.assertRaises(main.HTTPException) as raised:
            main.enforce_crustdata_search_scope(
                {"job_title": "Founder", "revenue_max": 100_000}, False
            )
        self.assertIn("revenue_max", raised.exception.detail)

    def test_profiles_are_deduped_by_crustdata_id(self):
        profiles = [
            {"crustdata_person_id": 1},
            {"crustdata_person_id": 1},
            {"crustdata_person_id": 2},
        ]
        self.assertEqual(
            [p["crustdata_person_id"] for p in main.dedupe_crustdata_profiles(profiles)],
            [1, 2],
        )


class FakeResponse:
    status_code = 200
    text = json.dumps({"profiles": [{"crustdata_person_id": 7}], "total_count": 12,
                       "next_cursor": "next-page"})

    def json(self):
        return json.loads(self.text)


class FakeClient:
    last_json = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers, json):
        self.__class__.last_json = json
        return FakeResponse()


class CrustdataRequestTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_semantic_search_cursor_sort_and_exclusions_are_sent(self):
        result = await main.crustdata_person_search(
            {"job_title": "Founder", "keywords": "outbound automation"},
            10,
            cursor="current-page",
            exclude_profiles=["https://linkedin.com/in/already-seen"],
        )
        body = FakeClient.last_json

        self.assertEqual(body["cursor"], "current-page")
        self.assertEqual(body["search"], {"query": "outbound automation", "mode": "hybrid"})
        self.assertEqual(body["mode"], "exact")
        self.assertEqual(
            body["post_processing"]["exclude_profiles"],
            ["https://linkedin.com/in/already-seen"],
        )
        self.assertEqual(result["next_cursor"], "next-page")
        self.assertEqual(result["total"], 12)

    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_sorts_are_omitted_when_the_search_is_semantic(self):
        """Crustdata 400s the whole request if sorts accompany a search query.

        Any ICP carrying keywords went semantic while still sending sorts, so
        Crustdata replied 400 "sorts are not supported when using semantic
        search" and the search failed outright — the caller then reported it as
        an exhausted credit balance.
        """
        await main.crustdata_person_search(
            {"job_title": "Founder", "company_size": "1-10",
             "industry": "computer software", "keywords": "pre-revenue"},
            10,
        )
        body = FakeClient.last_json

        self.assertNotIn("sorts", body)
        self.assertEqual(body["search"]["query"], "pre-revenue")

    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_sorts_are_sent_when_there_is_no_semantic_query(self):
        """Without a search query the stable sort still guards cursor paging."""
        await main.crustdata_person_search(
            {"job_title": "CEO", "location": "DE", "industry": "manufacturing"},
            10,
            cursor="page-2",
        )
        body = FakeClient.last_json

        self.assertEqual(body["sorts"], [{"field": "crustdata_person_id", "order": "asc"}])
        self.assertNotIn("search", body)
        self.assertEqual(body["cursor"], "page-2")


class ProviderStateTests(unittest.TestCase):
    """A missing Crustdata key must not silently bill searches to Wiza.

    Wiza's list workflow returns enriched contacts and spends an email credit
    per search. Since searching is free for the user now — the credit is spent
    on reveal — a silent degradation would turn every search into uncharged
    spend on the Wiza account.
    """

    def test_crustdata_with_a_key_is_not_degraded(self):
        with patch.object(main.settings, "search_provider", "crustdata"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", "key"):
            self.assertEqual(main.provider_state(), ("crustdata", False))
            self.assertFalse(main.provider_degraded())

    def test_missing_crustdata_key_degrades_and_is_flagged(self):
        with patch.object(main.settings, "search_provider", "crustdata"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", None):
            provider, degraded = main.provider_state()
            self.assertEqual(provider, "wiza")
            self.assertTrue(degraded)

    def test_deliberate_wiza_is_not_treated_as_degraded(self):
        # Choosing Wiza on purpose keeps the full enrichment workflow; only the
        # accidental fallback is routed to the credit-free preview.
        with patch.object(main.settings, "search_provider", "wiza"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", None):
            self.assertEqual(main.provider_state(), ("wiza", False))


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_reports_the_active_provider(self):
        with patch.object(main.settings, "search_provider", "crustdata"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", "key"):
            body = await main.health_check()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["search_provider"], "crustdata")
        self.assertFalse(body["degraded"])
        self.assertIsNone(body["degraded_reason"])

    async def test_health_surfaces_a_degraded_provider(self):
        with patch.object(main.settings, "search_provider", "crustdata"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", None):
            body = await main.health_check()
        self.assertEqual(body["search_provider"], "wiza")
        self.assertTrue(body["degraded"])
        self.assertIn("crustdata", body["degraded_reason"])
        self.assertIn("wiza", body["degraded_reason"])


class ProviderChainTests(unittest.TestCase):
    """Bytemine leads; Crustdata and Wiza follow as fallbacks."""

    def chain(self, **overrides):
        defaults = {"search_provider": "bytemine", "bytemine_api_key": "b",
                    "crustdata_api_key": "c", "wiza_api_key": "w"}
        defaults.update(overrides)
        patches = [patch.object(main.settings, k, v) for k, v in defaults.items()]
        for p in patches:
            p.start()
        try:
            return main.provider_chain()
        finally:
            for p in patches:
                p.stop()

    def test_all_three_configured_run_in_capability_order(self):
        self.assertEqual(self.chain(), ["bytemine", "crustdata", "wiza"])

    def test_a_provider_without_a_key_drops_out_of_the_chain(self):
        self.assertEqual(self.chain(bytemine_api_key=None), ["crustdata", "wiza"])
        self.assertEqual(
            self.chain(bytemine_api_key=None, crustdata_api_key=None), ["wiza"])

    def test_search_provider_pins_the_lead_and_the_rest_still_follow(self):
        # Pinning one provider must not discard the others: they are what makes
        # a failure recoverable.
        self.assertEqual(self.chain(search_provider="wiza"),
                         ["wiza", "bytemine", "crustdata"])
        self.assertEqual(self.chain(search_provider="crustdata"),
                         ["crustdata", "bytemine", "wiza"])

    def test_an_unknown_provider_name_does_not_empty_the_chain(self):
        self.assertEqual(self.chain(search_provider="nonesuch"),
                         ["bytemine", "crustdata", "wiza"])


class TregProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_treg_joins_the_waterfall_when_configured(self):
        with patch.object(main.settings, "treg_token", "treg-token"), \
             patch.object(main.settings, "bytemine_api_key", None), \
             patch.object(main.settings, "crustdata_api_key", None), \
             patch.object(main.settings, "getleads_api_key", None), \
             patch.object(main.settings, "coldiq_api_key", "coldiq"), \
             patch.object(main.settings, "wiza_api_key", "wiza"):
            self.assertEqual(main.provider_chain(), ["treg", "coldiq", "wiza"])

    def test_search_payload_uses_only_the_lead_gen_route(self):
        payload = main.build_treg_people_search({
            "job_title": "VP Sales",
            "company_domain": "acme.com",
            "location": "New York",
            "industry": "software",
            "limit": 5,
        }, 5)
        self.assertEqual(payload["title"], "VP Sales")
        self.assertEqual(payload["company_domain"], "acme.com")
        self.assertEqual(payload["keywords"], ["software"])
        self.assertEqual(payload["limit"], 5)

    def test_company_size_uses_an_exact_firmographic_treg_route(self):
        endpoint, payload = main.treg_search_plan({
            "company_size": "51-200",
            "job_title": "VP Sales",
        }, 10)
        self.assertEqual(endpoint, "leadsforge.people.search")
        self.assertEqual(
            payload["companyEmployeeNumberRange"], {"min": 51, "max": 200})
        self.assertEqual(payload["leadJobTitles"], {"include": ["VP Sales"]})

    def test_leadsforge_rows_keep_nested_firmographics(self):
        lead = main.transform_treg_person({
            "firstName": "Ada",
            "lastName": "Lovelace",
            "jobTitle": "VP Engineering",
            "location": {"city": "London", "country": "United Kingdom"},
            "company": {
                "name": "Analytical Engines",
                "domain": "analytical.example",
                "employeeCount": 120,
                "industry": "software development",
            },
        })
        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["company_size"], "51-200")
        self.assertEqual(lead["company_headcount"], 120)
        self.assertEqual(lead["location"], "London, United Kingdom")

    async def test_firmographic_response_and_cursor_are_read(self):
        with patch.object(main, "treg_call", new=AsyncMock(return_value={
            "leads": [{"firstName": "Ada"}], "cursor": "next-page",
        })) as call:
            result = await main.treg_person_search(
                {"company_size": "51-200"}, 10, cursor="current-page")
        self.assertEqual(result["profiles"], [{"firstName": "Ada"}])
        self.assertEqual(result["next_cursor"], "next-page")
        self.assertEqual(call.await_args.args[0], "leadsforge.people.search")
        self.assertEqual(call.await_args.kwargs["query"], {"cursor": "current-page"})

    async def test_every_call_is_tagged_metered_and_recorded(self):
        class Response:
            is_error = False
            headers = {
                "X-Treg-Call-Id": "call_123",
                "X-Treg-Cost-Micro": "4200",
                "X-Treg-Served-By": "icypeas",
            }

            def json(self):
                return {"output": {"people": []}}

        class Client:
            last_headers = None
            last_json = None

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, headers, json, params=None):
                self.__class__.last_headers = headers
                self.__class__.last_json = json
                return Response()

        context_token = main._treg_request_context.set({
            "customer_id": "cust_8123",
            "workspace_id": "ws_9",
            "idempotency_key": "retry-1",
        })
        try:
            with patch.object(main.settings, "treg_token", "secret"), \
                 patch.object(main.httpx, "AsyncClient", Client), \
                 patch.object(main, "record_treg_usage", new=AsyncMock()) as record:
                await main.treg_call("treg.people.search", {"limit": 1}, "lead-search")
        finally:
            main._treg_request_context.reset(context_token)

        self.assertEqual(
            Client.last_headers["X-Treg-Meta"],
            "customer=cust_8123, workspace=ws_9, feature=lead-search",
        )
        self.assertEqual(Client.last_headers["Idempotency-Key"], "retry-1")
        record.assert_awaited_once()
        self.assertEqual(record.await_args.kwargs["call_id"], "call_123")
        self.assertEqual(record.await_args.kwargs["cost_micro"], 4200)

    async def test_missing_customer_never_creates_unattributed_spend(self):
        context_token = main._treg_request_context.set({})
        try:
            with patch.object(main.settings, "treg_token", "secret"):
                with self.assertRaises(HTTPException) as raised:
                    await main.treg_call("treg.people.search", {"limit": 1}, "lead-search")
        finally:
            main._treg_request_context.reset(context_token)
        self.assertEqual(raised.exception.status_code, 400)

    def test_explicit_customer_header_wins(self):
        request = main.Request({
            "type": "http", "method": "GET", "path": "/",
            "headers": [(b"x-customer-id", b"cust_8123")],
            "client": ("127.0.0.1", 1234),
        })
        self.assertEqual(main.treg_customer_id_from_request(request), "cust_8123")

    def test_authenticated_customer_is_stable_without_custom_header(self):
        claims = base64.urlsafe_b64encode(
            json.dumps({"sub": "user-42"}).encode()).decode().rstrip("=")
        token = f"ignored.{claims}.ignored"
        request = main.Request({
            "type": "http", "method": "GET", "path": "/",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "client": ("127.0.0.1", 1234),
        })
        customer = main.treg_customer_id_from_request(request)
        self.assertRegex(customer, r"^auth_[0-9a-f]{24}$")
        self.assertEqual(customer, main.treg_customer_id_from_request(request))

    def test_anonymous_request_gets_pseudonymous_attribution(self):
        request = main.Request({
            "type": "http", "method": "GET", "path": "/",
            "headers": [(b"user-agent", b"salesos-test")],
            "client": ("203.0.113.7", 1234),
        })
        with patch.object(main.settings, "treg_default_customer_id", None):
            customer = main.treg_customer_id_from_request(request)
        self.assertRegex(customer, r"^anon_[0-9a-f]{24}$")
        self.assertNotIn("203.0.113.7", customer)

    async def test_invoice_usage_comes_from_the_reconciled_treg_ledger(self):
        class Response:
            is_error = False
            headers = {}

            def json(self):
                return {
                    "rows": [{
                        "value": "cust_8123", "charged_micro": 41234,
                        "charged_usd": 0.041234, "calls": 22,
                    }],
                    "attributed_micro": 41234,
                    "unattributed_micro": 1880,
                    "total_micro": 43114,
                }

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, headers, params):
                self.params = params
                return Response()

        with patch.object(main.settings, "treg_token", "secret"), \
             patch.object(main.settings, "treg_org_id", "org_1"), \
             patch.object(main.settings, "billing_admin_key", "admin"), \
             patch.object(main.httpx, "AsyncClient", Client):
            result = await main.treg_customer_usage(
                "cust_8123", days=30, x_billing_admin_key="admin")

        self.assertTrue(result["ledger_reconciled"])
        self.assertEqual(result["customer"]["charged_micro"], 41234)
        self.assertTrue(result["unattributed_warning"])

    def test_team_balance_details_are_never_exposed(self):
        class Response:
            status_code = 402
            headers = {"X-Treg-Error": "1"}

            def json(self):
                return {
                    "error": "insufficient_balance",
                    "balance_micro": 12,
                    "topup_url": "https://private.example/top-up",
                }

        error = main._safe_treg_error(Response())
        self.assertEqual(error.status_code, 503)
        self.assertNotIn("balance", str(error.detail))
        self.assertNotIn("top-up", str(error.detail))

class BytemineFilterTests(unittest.TestCase):
    def test_core_filters_map_onto_bytemine_fields(self):
        body = main.build_bytemine_filters({
            # Deliberately a title that implies no seniority, so the
            # seniority mapping below is actually exercised — see
            # SeniorityImpliedByTitleTests for the title-implies-it case.
            "job_title": "Account Executive",
            "seniority": "vp",
            "industry": "Information Technology and Services",
            "company_size": "51-200",
            "company_domain": "stripe.com",
            # A state by name. "CA" would be Canada here — see
            # classify_location — and Bytemine has no country field.
            "location": "California",
        })

        self.assertEqual(body["jobTitles"], ["Account Executive"])
        self.assertEqual(body["seniorityLevels"], ["VP"])
        self.assertEqual(body["industries"], ["Information Technology and Services"])
        self.assertEqual(body["employeeSizes"], ["51-200"])
        self.assertEqual(body["urls"], ["stripe.com"])
        self.assertEqual(body["states"], ["CA"])

    def test_founder_maps_to_the_owner_seniority(self):
        # No job title, so the mapping is what is under test rather than the
        # implied-seniority skip.
        body = main.build_bytemine_filters({"seniority": "founder"})
        self.assertEqual(body["seniorityLevels"], ["Owner"])

    def test_c_suite_maps_to_the_canonical_c_team_spelling(self):
        for value in ("cxo", "c-level", "c-suite", "executive"):
            body = main.build_bytemine_filters(
                {"job_title": "Account Executive", "seniority": value})
            self.assertEqual(body["seniorityLevels"], ["C-Team"], value)

    def test_a_city_goes_to_cities_rather_than_states(self):
        body = main.build_bytemine_filters({"job_title": "CEO", "location": "San Francisco"})
        self.assertEqual(body["cities"], ["San Francisco"])
        self.assertNotIn("states", body)

    def test_a_country_is_refused_so_the_chain_moves_on(self):
        # /contacts/search has no country field. Dropping the filter would search
        # the whole world under a location the user set, so the provider declines
        # and Crustdata — which does support country — takes the query.
        for country in ("US", "USA", "GB"):
            with self.assertRaises(main.ProviderUnsupported) as ctx:
                main.build_bytemine_filters({"job_title": "CEO", "location": country})
            self.assertEqual(ctx.exception.field, "location")

    def test_keywords_are_refused_rather_than_dropped(self):
        with self.assertRaises(main.ProviderUnsupported) as ctx:
            main.build_bytemine_filters({"job_title": "Founder", "keywords": "pre-revenue"})
        self.assertEqual(ctx.exception.field, "keywords")

    def test_an_empty_icp_is_a_client_error_not_a_fallback(self):
        with self.assertRaises(HTTPException) as ctx:
            main.build_bytemine_filters({})
        self.assertEqual(ctx.exception.status_code, 400)


class BytemineEmployeeBandTests(unittest.TestCase):
    def test_ranges_land_in_the_band_holding_their_floor(self):
        self.assertEqual(main.bytemine_employee_band("1-10"), "1-10")
        self.assertEqual(main.bytemine_employee_band("2-10"), "1-10")
        self.assertEqual(main.bytemine_employee_band("51-200"), "51-200")
        self.assertEqual(main.bytemine_employee_band("201-500"), "201-500")

    def test_the_top_band_uses_bytemines_spelling(self):
        # Bytemine's top band is "10000+", not the "10001+" Wiza uses. The wrong
        # string filters nothing at all.
        self.assertEqual(main.bytemine_employee_band("20000+"), "10000+")

    def test_a_size_with_no_number_has_no_band(self):
        self.assertIsNone(main.bytemine_employee_band("enterprise"))
        self.assertIsNone(main.bytemine_employee_band(""))


class BytemineTransformTests(unittest.TestCase):
    SEARCH_RESULT = {
        "pid": "10000000331",
        "first_name": "Ada",
        "last_name": "Lovelace",
        "job_title": "VP of Sales",
        "department": "Sales",
        "company_name": "Acme",
        "company_domain": "acme.com",
        "company_industry": "Computer Software",
        "company_employee_range": "51-200",
        "company_revenue_range": "$10M-$50M",
        "city": "San Francisco",
        "state": "CA",
        "linkedin_url": "https://linkedin.com/in/ada",
        "email": "***",
        "phone": "***",
    }

    def test_search_results_expose_availability_not_values(self):
        lead = main.transform_bytemine_profile(self.SEARCH_RESULT)

        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["job_title"], "VP of Sales")
        self.assertEqual(lead["company_domain"], "acme.com")
        self.assertEqual(lead["location"], "San Francisco, CA")
        # Masked in search: the value costs a credit and comes from the unlock.
        self.assertIsNone(lead["business_email"])
        self.assertIsNone(lead["phone"])
        self.assertTrue(lead["email_available"])
        self.assertTrue(lead["phone_available"])

    def test_the_pid_is_carried_so_the_reveal_hits_the_same_record(self):
        lead = main.transform_bytemine_profile(self.SEARCH_RESULT)
        self.assertEqual(lead["bytemine_pid"], "10000000331")

    def test_a_contact_without_an_email_is_not_reported_as_available(self):
        lead = main.transform_bytemine_profile({**self.SEARCH_RESULT, "email": ""})
        self.assertFalse(lead["email_available"])
        self.assertIsNone(lead["email_status"])

    def test_unlocked_records_carry_the_real_values(self):
        lead = main.transform_bytemine_unlocked({
            **self.SEARCH_RESULT,
            "full_name": "Ada Lovelace",
            "work_email": "ada@acme.com",
            "phone": "(415) 555-1234",
            "linkedin_profile": "https://linkedin.com/in/ada",
        })

        self.assertEqual(lead["business_email"], "ada@acme.com")
        self.assertEqual(lead["email_status"], "verified")
        self.assertEqual(lead["phone"], "(415) 555-1234")
        self.assertTrue(lead["email_available"])

    def test_a_still_masked_value_is_not_mistaken_for_a_real_one(self):
        lead = main.transform_bytemine_unlocked({**self.SEARCH_RESULT})
        self.assertIsNone(lead["business_email"])
        self.assertEqual(lead["email_status"], "no_email")


class BytemineRequestTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_the_gateway_envelope_carries_the_path_and_body(self):
        # Every Bytemine endpoint is POSTed to one URL with the real path and
        # method inside the JSON body.
        with patch.object(main.settings, "bytemine_api_key", "key"):
            FakeResponse.text = json.dumps({
                "data": [{"pid": "1", "first_name": "Ada"}],
                "pagination": {"total": 7, "has_more": False},
                "credits_used": 1,
            })
            try:
                result = await main.bytemine_person_search(
                    {"job_title": "VP of Sales", "location": "California"}, 25)
            finally:
                FakeResponse.text = json.dumps({
                    "profiles": [{"crustdata_person_id": 7}],
                    "total_count": 12, "next_cursor": "next-page"})

        envelope = FakeClient.last_json
        self.assertEqual(envelope["path"], "/contacts/search")
        self.assertEqual(envelope["method"], "POST")
        self.assertEqual(envelope["body"]["jobTitles"], ["VP of Sales"])
        self.assertEqual(envelope["body"]["states"], ["CA"])
        self.assertEqual(envelope["body"]["pageSize"], 25)
        self.assertEqual(result["total"], 7)
        self.assertEqual(len(result["profiles"]), 1)


class ProviderFallbackTests(unittest.IsolatedAsyncioTestCase):
    """The chain is only worth having if a search actually moves down it."""

    async def run_search(self, request, responses, crustdata_key="c"):
        """Run /search with the cache disabled and each provider call stubbed.

        `responses` maps a provider name to either the value its fetcher should
        return or an exception it should raise. A provider left out of the map
        is not stubbed at all, so its real implementation runs — which is how the
        genuine "this provider cannot express that filter" refusal gets covered.

        `crustdata_key=None` drops Crustdata from the chain, which is how a
        request that no configured provider can express gets covered: Crustdata
        is the only one of the three with a free-text field.
        """
        calls: list = []

        def stub(name):
            async def run(*args, **kwargs):
                calls.append(name)
                r = responses[name]
                if isinstance(r, Exception):
                    raise r
                return r
            return run

        async def no_cache(_hash):
            return None

        async def no_store(*args, **kwargs):
            return None

        patches = [
            patch.object(main.settings, "bytemine_api_key", "b"),
            patch.object(main.settings, "crustdata_api_key", crustdata_key),
            patch.object(main.settings, "search_provider", "bytemine"),
            patch.object(main, "cache_lookup", no_cache),
            patch.object(main, "cache_store", no_store),
        ]
        for name, attr in (("bytemine", "bytemine_person_search"),
                           ("crustdata", "crustdata_person_search"),
                           ("wiza", "fetch_from_wiza")):
            if name in responses:
                patches.append(patch.object(main, attr, stub(name)))

        for p in patches:
            p.start()
        try:
            response = await main.search_leads(request)
        finally:
            for p in patches:
                p.stop()
        return response, calls

    async def test_a_typed_query_uses_the_first_successful_provider(self):
        """The regression the logs caught.

        Every search from the UI carries the user's sentence, and both Bytemine
        and Wiza were refusing on it, so a three-provider chain ran one
        provider: `bytemine:unsupported_filter, crustdata:N,
        wiza:unsupported_filter` on every line.
        """
        response, calls = await self.run_search(
            main.SearchRequest(query="Founders at AI SaaS companies with a size of 1-10",
                               job_title="founder", industry="computer software"),
            {"bytemine": {"profiles": [{"pid": "1", "first_name": "Ada"}],
                          "total": 1, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None},
             "wiza": [{"full_name": "Alan T", "company": "Bletchley"}]},
        )

        # The point of this test is the refusal, not the stopping: neither
        # Bytemine nor Wiza may sit out a search just because it carries the
        # user's sentence.
        outcomes = {a["provider"]: a["outcome"] for a in response.provider_attempts}
        self.assertNotIn("unsupported_filter", outcomes.values())
        self.assertEqual(calls[0], "bytemine")
        self.assertGreaterEqual(response.count, 1)

    async def test_a_full_page_stops_the_waterfall(self):
        # Nothing below is called speculatively: the page is full, so the walk
        # ends and the lower-priority tools are never billed.
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA", limit=2),
            {"bytemine": {"profiles": [{"pid": "1", "first_name": "Ada"},
                                       {"pid": "2", "first_name": "Grace"}],
                          "total": 2, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(calls, ["bytemine"])
        self.assertEqual(response.provider, "bytemine")
        self.assertEqual(response.count, 2)
        self.assertIn("Ada", [l["contact_name"] for l in response.leads])

    async def test_a_partial_page_is_topped_up_by_the_next_provider(self):
        """One lead used to end a search for ten.

        `if found: break` stopped on the first leg with any row at all, so a
        single Bytemine result ended the chain and the other providers were
        never asked — the user saw a nearly empty page while the tools that
        could have filled it sat idle. That is the single-provider dependency a
        chain exists to prevent.
        """
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA", limit=3),
            {"bytemine": {"profiles": [{"pid": "1", "first_name": "Ada"}],
                          "total": 1, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(calls, ["bytemine", "crustdata"])
        self.assertEqual(response.count, 2)
        self.assertEqual(response.provider, "bytemine+crustdata")

    async def test_a_later_leg_is_asked_only_for_the_shortfall(self):
        # Several of these bill per record returned, so a leg topping up a page
        # must not re-request the whole page.
        asked: list = []

        async def crustdata(params, limit, **kwargs):
            asked.append(limit)
            return {"profiles": [{"crustdata_person_id": 9}], "total": 1,
                    "next_cursor": None}

        with patch.object(main, "crustdata_person_search", crustdata):
            await self.run_search(
                main.SearchRequest(job_title="VP of Sales", location="CA", limit=5),
                {"bytemine": {"profiles": [{"pid": "1", "first_name": "Ada"},
                                           {"pid": "2", "first_name": "Grace"}],
                              "total": 2, "next_cursor": None}},
            )

        self.assertEqual(asked, [3])

    async def test_a_provider_with_nothing_does_not_decide_the_search(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": {"profiles": [], "total": 0, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(response.provider, "crustdata")
        self.assertEqual(response.count, 1)
        outcomes = {a["provider"]: a["outcome"] for a in response.provider_attempts}
        self.assertEqual(outcomes["bytemine"], "no_results")

    async def test_a_filter_bytemine_cannot_express_moves_to_crustdata(self):
        # A country location has no field on /contacts/search. Falling through
        # keeps the ICP whole; dropping the country would not.
        response, calls = await self.run_search(
            main.SearchRequest(job_title="Founder", location="US"),
            # Bytemine is deliberately not stubbed: the real
            # build_bytemine_filters must be the thing that refuses.
            {"crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        # Bytemine sits this one out; the providers that can express it answer.
        self.assertNotIn("bytemine", calls)
        self.assertEqual(response.provider, "crustdata")
        outcomes = {a["provider"]: a["outcome"] for a in response.provider_attempts}
        self.assertEqual(outcomes["bytemine"], "unsupported_filter")

    async def test_every_provider_refusing_is_an_error_not_an_empty_result(self):
        # Nothing ran, so "no leads" would be a different answer from the truth:
        # no configured provider can express the filter. An empty success here
        # reads as "no such people exist" and tells the user nothing to change.
        #
        # Bytemine refuses the country before spending a company-search credit;
        # Wiza refuses the keyword. Neither is stubbed, so both refusals are the
        # real ones, and bytemine_call must never be reached.
        with patch.object(main, "bytemine_call", self._fail("must not be called")):
            with self.assertRaises(HTTPException) as caught:
                await self.run_search(
                    main.SearchRequest(job_title="Founder", location="US",
                                       keywords="ai saas"),
                    {},
                    crustdata_key=None,
                )

        self.assertEqual(caught.exception.status_code, 422)
        detail = caught.exception.detail
        self.assertIn("keywords", detail)
        self.assertIn("location", detail)
        self.assertIn("credit was not used", detail)

    @staticmethod
    def _fail(message):
        async def run(*args, **kwargs):
            raise AssertionError(message)
        return run

    async def test_an_exhausted_bytemine_balance_falls_through(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": HTTPException(status_code=402, detail="no credits"),
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(sorted(calls), ["bytemine", "crustdata"])
        self.assertEqual(response.provider, "crustdata")
        details = {a["provider"]: a.get("detail") for a in response.provider_attempts}
        self.assertEqual(details["bytemine"], 402)

    async def test_no_results_from_the_leader_still_tries_the_next(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": {"profiles": [], "total": 0, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(sorted(calls), ["bytemine", "crustdata"])
        self.assertEqual(response.provider, "crustdata")
        outcomes = {a["provider"]: a["outcome"] for a in response.provider_attempts}
        self.assertEqual(outcomes["bytemine"], "no_results")

    async def test_a_provider_bad_request_falls_through(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": HTTPException(status_code=400, detail="bad filter"),
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                            "total": 1, "next_cursor": None}},
        )
        self.assertEqual(calls, ["bytemine", "crustdata"])
        self.assertEqual(response.provider, "crustdata")

    async def test_every_provider_failing_is_reported(self):
        # Nobody answered, so the error must reach the caller rather than being
        # swallowed into an empty result set that reads as "no such people".
        with self.assertRaises(HTTPException) as ctx:
            await self.run_search(
                main.SearchRequest(job_title="VP of Sales", location="CA"),
                {"bytemine": HTTPException(status_code=402, detail="no credits"),
                 "crustdata": HTTPException(status_code=500, detail="upstream down"),
                 "wiza": HTTPException(status_code=503, detail="wiza down")},
            )
        self.assertIn(ctx.exception.status_code, (402, 500, 503))


class WizaEmptyExportTests(unittest.TestCase):
    def test_no_contacts_export_is_a_zero_result_not_an_error(self):
        response = type("Response", (), {
            "status_code": 400,
            "text": '{"message":"No contacts to export."}',
        })()
        self.assertTrue(main.wiza_no_contacts(response))

    def test_other_400s_are_not_hidden(self):
        response = type("Response", (), {
            "status_code": 400,
            "text": '{"message":"Invalid list"}',
        })()
        self.assertFalse(main.wiza_no_contacts(response))


class PaginationTests(unittest.TestCase):
    """Page 2 must not be page 1.

    The caller sent `offset` and SearchRequest did not declare it, so pydantic
    dropped it: every page produced identical params, hashed to the same cache
    key, and was answered from page 1's cached rows. Every search looked like it
    held the same few leads forever.
    """

    def test_offset_survives_the_request_model(self):
        request = main.SearchRequest(**{"job_title": "CEO", "limit": 10, "offset": 10})
        self.assertEqual(request.start_offset, 10)

    def test_a_page_number_becomes_an_offset(self):
        self.assertEqual(main.SearchRequest(job_title="CEO", limit=10, page=1).start_offset, 0)
        self.assertEqual(main.SearchRequest(job_title="CEO", limit=10, page=3).start_offset, 20)
        self.assertEqual(main.SearchRequest(job_title="CEO", limit=25, page=2).start_offset, 25)

    def test_pages_are_no_longer_identical_requests(self):
        page1 = main.SearchRequest(**{"job_title": "CEO", "limit": 10})
        page2 = main.SearchRequest(**{"job_title": "CEO", "limit": 10, "offset": 10})
        self.assertNotEqual(page1.model_dump(), page2.model_dump())

    def test_an_explicit_offset_wins_over_the_page_number(self):
        request = main.SearchRequest(**{"job_title": "CEO", "limit": 10, "page": 5, "offset": 3})
        self.assertEqual(request.start_offset, 3)

    def test_a_nonsense_page_does_not_produce_a_negative_offset(self):
        self.assertEqual(main.SearchRequest(job_title="CEO", page=0).start_offset, 0)
        self.assertEqual(main.SearchRequest(job_title="CEO", page=-4).start_offset, 0)


class BytemineePaginationTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_an_offset_becomes_bytemines_zero_indexed_page(self):
        with patch.object(main.settings, "bytemine_api_key", "key"):
            FakeResponse.text = json.dumps({"data": [], "pagination": {"total": 0}})
            try:
                for offset, expected_page in ((0, 0), (10, 1), (25, 2), (100, 10)):
                    await main.bytemine_person_search(
                        {"job_title": "CEO", "location": "California"}, 10, offset=offset)
                    self.assertEqual(
                        FakeClient.last_json["body"]["page"], expected_page,
                        f"offset {offset}")
            finally:
                FakeResponse.text = json.dumps({
                    "profiles": [{"crustdata_person_id": 7}],
                    "total_count": 12, "next_cursor": "next-page"})


class IcpFilterRetentionTests(unittest.TestCase):
    """Wiza retries without a rejected filter; it must never drop the ICP.

    company_size, company_industry and company_location decide who the search is
    for. Retrying without one returns leads from the wrong size, industry or
    country while still reporting success.
    """

    def test_icp_defining_filters_are_not_droppable(self):
        for field in ("company_size", "company_industry", "company_location"):
            self.assertNotIn(field, main.DROPPABLE, field)

    def test_genuinely_optional_filters_remain_droppable(self):
        for field in ("job_title_level", "job_role", "skill", "funding_stage", "revenue"):
            self.assertIn(field, main.DROPPABLE, field)

    def test_the_icp_fields_are_named_for_the_error_message(self):
        self.assertEqual(
            set(main.ICP_DEFINING),
            {"company_size", "company_industry", "company_location"})


class SemanticTermTests(unittest.TestCase):
    """Words no provider taxonomy can express must survive the parser.

    Every provider classifies companies with LinkedIn's industry list, which has
    no AI category. Collapsing "AI SaaS" to `computer software` and stopping
    there searched every software company on earth, which is how an AI search
    came back with aerospace and venture capital.
    """

    def test_ai_terms_are_recognised(self):
        for text in ("AI SaaS", "ai saas", "generative AI companies",
                     "machine learning startups", "LLM infrastructure",
                     "AI-powered CRM"):
            self.assertTrue(main.semantic_terms_in(text), text)

    def test_words_that_merely_contain_ai_are_not_ai_queries(self):
        # \b would match the "ai" inside every one of these.
        for text in ("email marketing", "retail founders", "supply chain",
                     "chairman", "repair shops", "Dubai"):
            self.assertEqual(main.semantic_terms_in(text), [], text)

    def test_the_longest_term_wins(self):
        # Not ["ai"], which would send a weaker query than the user typed.
        self.assertEqual(main.semantic_terms_in("AI SaaS"), ["ai saas"])

    def test_the_defining_term_survives_as_a_keyword(self):
        filters = main.heuristic_parse_icp("AI SaaS founders")

        # The taxonomy value is kept as a supporting filter...
        self.assertEqual(filters["industry"], "computer software")
        # ...but the word that actually defines the ICP is searchable.
        self.assertEqual(filters["keywords"], "ai saas")

    def test_a_segment_after_at_is_not_a_company_name(self):
        # "VP Sales at AI SaaS startups" is not a request for a company called
        # "AI SaaS"; searching one as a company name matches near-nothing.
        filters = main.heuristic_parse_icp("VP Sales at AI SaaS startups")

        self.assertNotIn("company", filters)
        self.assertEqual(filters["keywords"], "ai saas")

    def test_an_ordinary_industry_query_gains_no_keyword(self):
        filters = main.heuristic_parse_icp("fintech CTOs in Germany")

        self.assertEqual(filters["industry"], "financial services")
        self.assertNotIn("keywords", filters)


class KeywordExpressibilityTests(unittest.TestCase):
    """A provider must search the free-text term or refuse the search."""

    def test_wiza_refuses_keywords_rather_than_widening_the_search(self):
        # Wiza ORs the values in a filter, so appending the keyword to job_title
        # matched every Founder anywhere — the opposite of narrowing.
        with self.assertRaises(main.ProviderUnsupported) as caught:
            main.build_wiza_filters({"job_title": "Founder", "keywords": "ai saas"})

        self.assertEqual(caught.exception.field, "keywords")

    def test_wiza_refuses_an_industry_it_cannot_map(self):
        # Wiza ignores an industry outside its vocabulary, so sending one and
        # carrying on silently searched every industry.
        with self.assertRaises(main.ProviderUnsupported) as caught:
            main.build_wiza_filters({"job_title": "Founder", "industry": "ai saas"})

        self.assertEqual(caught.exception.field, "industry")

    def test_wiza_still_accepts_a_mappable_industry(self):
        filters = main.build_wiza_filters({"industry": "artificial intelligence"})

        self.assertEqual(filters["company_industry"][0]["v"], "computer software")

    def test_bytemine_refuses_a_keyword_that_was_never_resolved(self):
        # Reaching the filter builder with a keyword still set means it would be
        # dropped — bytemine_person_search is meant to resolve it to domains.
        with self.assertRaises(main.ProviderUnsupported) as caught:
            main.build_bytemine_filters({"job_title": "Founder", "keywords": "ai saas"})

        self.assertEqual(caught.exception.field, "keywords")

    def test_bytemine_searches_resolved_companies_by_domain(self):
        filters = main.build_bytemine_filters({
            "job_title": "Founder",
            "company_domains": ["anthropic.com", "openai.com"],
        })

        self.assertEqual(filters["urls"], ["anthropic.com", "openai.com"])


class LocationClassificationTests(unittest.TestCase):
    """A location is a US state, a country or a city — never a guess.

    /contacts/search has no country field, and the old rule sent any token
    longer than two characters as a city. "Germany" became a search for a city
    called Germany and "Texas" a city called Texas: filters that match nothing,
    silently, while the search still reports success.
    """

    def test_a_country_is_recognised_as_one(self):
        for name in ("Germany", "United States", "Canada", "Japan"):
            self.assertEqual(main.classify_location(name)[0], "country", name)

    def test_a_state_name_becomes_its_code(self):
        self.assertEqual(main.classify_location("Texas"), ("state", "TX"))
        self.assertEqual(main.classify_location("New York"), ("state", "NY"))
        self.assertEqual(main.classify_location("Ohio"), ("state", "OH"))

    def test_an_unambiguous_state_code_is_kept(self):
        self.assertEqual(main.classify_location("TX"), ("state", "TX"))
        self.assertEqual(main.classify_location("NY"), ("state", "NY"))

    def test_a_code_naming_both_a_state_and_a_country_reads_as_the_country(self):
        # fetch-external-leads sets `location` from normalizeCountry(), so a
        # two-letter location is always a country code. Reading "CA" as
        # California sent every search for Canada to the wrong continent.
        #
        # Driven off the real set so a code added there cannot quietly go back
        # to being read as a US state.
        for code in main._STATE_CODE_IS_ALSO_A_COUNTRY:
            kind, value = main.classify_location(code)
            self.assertEqual((kind, value), ("country", code), code)

    def test_the_colliding_codes_really_are_us_state_codes(self):
        # If one of these stopped being a state code the entry would be dead
        # weight, and the ordering it justifies would look arbitrary.
        self.assertTrue(main._STATE_CODE_IS_ALSO_A_COUNTRY <= main._US_STATE_CODES)

    def test_a_state_name_still_beats_a_country_code_spelling(self):
        # "Georgia" the US state, not GE the country.
        self.assertEqual(main.classify_location("Georgia"), ("state", "GA"))

    def test_a_city_stays_a_city(self):
        self.assertEqual(main.classify_location("Berlin"), ("city", "Berlin"))
        self.assertEqual(main.classify_location("San Francisco"),
                         ("city", "San Francisco"))

    def test_a_country_is_refused_not_searched_as_a_city(self):
        with self.assertRaises(main.ProviderUnsupported) as caught:
            main.build_bytemine_filters({"job_title": "CEO", "location": "Germany"})

        self.assertEqual(caught.exception.field, "location")

    def test_a_state_reaches_bytemine_as_a_state(self):
        # Previously cities:["Texas"], which matches nothing.
        body = main.build_bytemine_filters({"job_title": "CEO", "location": "Texas"})

        self.assertEqual(body["states"], ["TX"])
        self.assertNotIn("cities", body)

    def test_a_city_still_reaches_bytemine_as_a_city(self):
        body = main.build_bytemine_filters({"job_title": "CEO", "location": "Berlin"})

        self.assertEqual(body["cities"], ["Berlin"])

    def test_the_parser_state_names_are_no_longer_shadowed(self):
        # A second _US_STATES of 2-letter codes used to overwrite the parser's
        # full-name set at module scope, so the names were unreachable.
        self.assertIn("california", main._US_STATES)
        self.assertIn("CA", main._US_STATE_CODES)


class BytemineKeywordResolutionTests(unittest.IsolatedAsyncioTestCase):
    """Bytemine carries a free-text term via the company graph.

    /contacts/search has no free-text field, but /b2b-search matches company
    descriptions — which is where a segment like "AI SaaS" is written down.
    """

    def setUp(self):
        self.calls = []

    def gateway(self, companies):
        async def call(path, body, timeout=60.0):
            self.calls.append((path, body))
            if path == "/b2b-search":
                return {"data": companies}
            return {"data": [{"pid": "1"}], "pagination": {"total": 1}}
        return call

    async def test_a_keyword_becomes_a_company_domain_filter(self):
        with patch.object(main, "bytemine_call", self.gateway(
                [{"website": "https://anthropic.com"}, {"website": "openai.com"}])):
            await main.bytemine_person_search(
                {"keywords": "ai saas", "job_title": "Founder"}, 10)

        paths = [path for path, _ in self.calls]
        self.assertEqual(paths, ["/b2b-search", "/contacts/search"])
        self.assertEqual(self.calls[0][1]["keywords"], "ai saas")
        self.assertEqual(self.calls[1][1]["urls"], ["anthropic.com", "openai.com"])
        # The term is spent on the domain filter, never sent as a raw keyword.
        self.assertNotIn("keywords", self.calls[1][1])

    async def test_a_keyword_matching_no_company_returns_no_leads(self):
        # Searching on without the term would answer a different question.
        with patch.object(main, "bytemine_call", self.gateway([])):
            result = await main.bytemine_person_search(
                {"keywords": "quantum abacus", "job_title": "CEO"}, 10)

        self.assertEqual(result["profiles"], [])
        self.assertEqual([path for path, _ in self.calls], ["/b2b-search"])

    async def test_a_company_without_a_website_cannot_be_used(self):
        with patch.object(main, "bytemine_call", self.gateway(
                [{"name": "No Site"}, {"website": "scale.com"}])):
            await main.bytemine_person_search({"keywords": "ai"}, 10)

        self.assertEqual(self.calls[1][1]["urls"], ["scale.com"])

    async def test_an_unexpressible_filter_is_refused_before_spending_credits(self):
        # /b2b-search bills per company returned; a country location is going to
        # be refused anyway, so the request must not reach the gateway.
        async def must_not_call(*args, **kwargs):
            raise AssertionError("gateway reached for a refused search")

        with patch.object(main, "bytemine_call", must_not_call):
            with self.assertRaises(main.ProviderUnsupported):
                await main.bytemine_person_search(
                    {"keywords": "ai saas", "location": "US"}, 10)


class SemanticQueryRetentionTests(unittest.IsolatedAsyncioTestCase):
    """The sentence the user typed must survive into the search.

    resolve_search_params popped `query` and then only used it when no
    structured field was set — which is never, because the frontend parses the
    query into filters before sending and ships both. So the words were dropped,
    and every search with the same coarse filters became the same request:
    identical params, identical cache key, identical rows for the cache's whole
    hour. Different searches returned the same leads because by that point they
    were the same search.
    """

    async def resolve(self, **kwargs):
        return await main.resolve_search_params(main.SearchRequest(**kwargs))

    async def test_the_sentence_survives_alongside_structured_filters(self):
        params = await self.resolve(
            query="heads of RevOps at fintechs replacing Salesforce",
            job_title="VP Sales", industry="computer software", location="US")

        self.assertEqual(params["semantic_query"],
                         "heads of RevOps at fintechs replacing Salesforce")

    async def test_two_different_sentences_are_two_different_searches(self):
        common = {"job_title": "VP Sales", "industry": "computer software", "location": "US"}
        a = await self.resolve(query="VP Sales at AI SaaS companies", **common)
        b = await self.resolve(query="heads of RevOps at fintechs", **common)

        self.assertNotEqual(main.generate_search_hash(a), main.generate_search_hash(b))

    async def test_a_bare_query_still_gets_parsed_into_filters(self):
        # The no-structured-fields path is unchanged: the parser runs and the
        # sentence becomes filters rather than only a ranking hint.
        params = await self.resolve(query="fintech CTOs in Germany")

        self.assertEqual(params["industry"], "financial services")
        self.assertEqual(params["job_title"], "CTO")

    def test_crustdata_searches_the_sentence(self):
        query = "heads of RevOps at fintechs replacing Salesforce"
        joined = " ".join(x for x in ("", query) if x).strip()

        self.assertEqual(joined, query)

    def test_providers_without_free_text_still_search(self):
        # The sentence must not sideline a provider. The structured filters it
        # was parsed into are in the body already, so a provider that cannot
        # rank by the phrasing still contributes what those filters find.
        #
        # Refusing here meant Bytemine and Wiza sat out every search from the
        # UI — which always sends a query — so a three-provider chain quietly
        # ran one provider.
        params = {"job_title": "VP Sales", "location": "Texas",
                  "semantic_query": "heads of RevOps at fintechs"}

        bytemine = main.build_bytemine_filters(params)
        wiza = main.build_wiza_filters(params)

        self.assertEqual(bytemine["jobTitles"], ["VP Sales"])
        self.assertEqual(bytemine["states"], ["TX"])
        self.assertNotIn("semantic_query", bytemine)
        self.assertEqual(wiza["job_title"][0]["v"], "VP Sales")
        self.assertNotIn("semantic_query", wiza)

    def test_a_keyword_filter_is_still_refused(self):
        # keywords are a criterion the user stated, unlike the sentence the
        # filters were derived from. Dropping one silently would search for
        # something the user did not ask for.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_wiza_filters({"job_title": "VP Sales", "keywords": "ai saas"})


class ColdiqFilterTests(unittest.TestCase):
    """ColdIQ finds decision-makers at companies, and says so.

    Its FindPeopleInput has company_domains, company_linkedin_urls, job_titles,
    seniorities, locations, keywords, limit and max_per_company. There is no
    industry field and no headcount field, so an ICP stated in those terms is
    not expressible and must be refused rather than answered with whoever
    matched the remaining filters.
    """

    def test_core_filters_map_onto_coldiq_fields(self):
        body = main.build_coldiq_filters({
            "job_title": "Account Executive",
            "seniority": "vp",
            "company_domain": "https://www.stripe.com/pricing",
            "location": "United States",
        })

        self.assertEqual(body["job_titles"], ["Account Executive"])
        self.assertEqual(body["seniorities"], ["vp"])
        self.assertEqual(body["company_domains"], ["stripe.com"])
        self.assertEqual(body["locations"], ["US"])

    def test_our_seniorities_map_onto_coldiqs_vocabulary(self):
        for ours, theirs in (("cxo", "c_suite"), ("c-suite", "c_suite"),
                             ("founder", "owner"), ("vp", "vp")):
            body = main.build_coldiq_filters({"job_title": "X", "seniority": ours})
            self.assertEqual(body["seniorities"], [theirs], ours)

    def test_an_unmapped_seniority_is_left_out_rather_than_guessed(self):
        body = main.build_coldiq_filters({"job_title": "X", "seniority": "grand poobah"})

        self.assertNotIn("seniorities", body)
        self.assertEqual(body["job_titles"], ["X"])

    def test_industry_becomes_a_keyword_hint_rather_than_a_refusal(self):
        # FindPeopleInput has no industry field. Refusing outright made ColdIQ
        # inert — nearly every real search names an industry, so it sat out all
        # of them and contributed nothing. It goes in as a ranking hint, and the
        # rows say which dimensions were only ranked.
        body = main.build_coldiq_filters(
            {"job_title": "Founder", "industry": "computer software"})

        self.assertEqual(body["job_titles"], ["Founder"])
        self.assertIn("computer software", body["keywords"])
        self.assertEqual(body["_unverified_dimensions"], ["industry"])

    def test_company_size_becomes_a_keyword_hint_too(self):
        body = main.build_coldiq_filters(
            {"job_title": "Founder", "company_size": "1-10"})

        self.assertIn("1-10 employees", body["keywords"])
        self.assertEqual(body["_unverified_dimensions"], ["company_size"])

    def test_a_search_with_nothing_exact_to_anchor_it_is_still_refused(self):
        # Keywords alone are a hint, not a constraint: with no title, seniority,
        # location or domain, ColdIQ's waterfall would return whoever it liked.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_coldiq_filters({"industry": "computer software"})

    def test_the_marker_never_reaches_the_api(self):
        # _unverified_dimensions is ours, not a FindPeopleInput field.
        body = main.build_coldiq_filters(
            {"job_title": "Founder", "industry": "computer software"})
        sent = {k: v for k, v in body.items() if not k.startswith("_")}

        self.assertNotIn("_unverified_dimensions", sent)
        self.assertIn("job_titles", sent)

    def test_a_sub_national_location_is_refused_not_sent_as_a_country(self):
        # `locations` takes ISO-2 codes or country names. "Texas" would be read
        # as a country name and match nothing, which is a silent miss.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_coldiq_filters({"job_title": "VP Sales", "location": "Texas"})

    def test_a_bare_company_name_is_refused(self):
        # ColdIQ keys on domain or LinkedIn company URL, not a name.
        with self.assertRaises(main.ProviderUnsupported) as caught:
            main.build_coldiq_filters({"job_title": "VP Sales", "company": "Acme Corp"})

        self.assertEqual(caught.exception.field, "company")

    def test_a_company_value_that_is_really_a_domain_is_used_as_one(self):
        body = main.build_coldiq_filters({"company": "workflows.io"})

        self.assertEqual(body["company_domains"], ["workflows.io"])


class ColdiqTransformTests(unittest.TestCase):
    """The record shape is not in ColdIQ's spec, so the reader is tolerant.

    /v1/people/search declares `data: nullable` described only as "Normalized,
    provider-agnostic result". Every spelling ColdIQ uses for the same idea
    elsewhere in its own schemas is accepted, and the search logs the first
    record's keys so the real shape is one production search away.
    """

    def test_snake_case_record(self):
        lead = main.transform_coldiq_profile({
            "first_name": "Ada", "last_name": "Lovelace",
            "job_title": "VP of Sales", "company_name": "Acme",
            "domain": "acme.com", "email": "ada@acme.com",
            "linkedin_url": "https://linkedin.com/in/ada",
        })

        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["company_domain"], "acme.com")
        self.assertEqual(lead["business_email"], "ada@acme.com")
        self.assertEqual(lead["provider"], "coldiq")

    def test_camel_case_record(self):
        lead = main.transform_coldiq_profile({
            "fullName": "Grace Hopper", "jobTitle": "CTO",
            "companyName": "Navy", "linkedinUrl": "https://linkedin.com/in/grace",
        })

        self.assertEqual(lead["contact_name"], "Grace Hopper")
        self.assertEqual(lead["job_title"], "CTO")
        self.assertEqual(lead["company_name"], "Navy")

    def test_a_nested_company_object(self):
        lead = main.transform_coldiq_profile({
            "name": "Alan Turing", "title": "Lead",
            "company": {"name": "Bletchley", "domain": "bletchley.uk"},
        })

        self.assertEqual(lead["company_name"], "Bletchley")

    def test_an_unrecognised_record_yields_nulls_rather_than_junk(self):
        # A record we cannot read must not become a lead with a name like
        # "None None" — the merge dedupes on name+company as a last resort.
        lead = main.transform_coldiq_profile({"unexpected": "shape"})

        self.assertIsNone(lead["contact_name"])
        self.assertIsNone(lead["company_name"])


class ColdiqRequestTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main.httpx, "AsyncClient", FakeClient)
    async def test_the_search_body_matches_coldiqs_envelope(self):
        with patch.object(main.settings, "coldiq_api_key", "key"):
            FakeResponse.text = json.dumps({
                "data": [{"first_name": "Ada", "company_name": "Acme"}],
                "_meta": {"provider": "prospeo", "credits": 1},
            })
            try:
                result = await main.coldiq_person_search(
                    {"job_title": "VP of Sales", "company_domain": "acme.com"}, 25)
            finally:
                FakeResponse.text = json.dumps({
                    "profiles": [{"crustdata_person_id": 7}],
                    "total_count": 12, "next_cursor": "next-page"})

        body = FakeClient.last_json
        self.assertEqual(body["input"]["job_titles"], ["VP of Sales"])
        self.assertEqual(body["input"]["company_domains"], ["acme.com"])
        self.assertEqual(body["input"]["limit"], 25)
        # auto uses ColdIQ's own waterfall; pinning one vendor would make this
        # leg a worse copy of a provider we already call directly.
        self.assertEqual(body["provider"], "auto")
        self.assertEqual(body["fields"], "compact")
        self.assertEqual(len(result["profiles"]), 1)


class ColdiqChainTests(unittest.TestCase):
    def test_coldiq_joins_the_chain_when_configured(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "crustdata_api_key", "c"), \
             patch.object(main.settings, "coldiq_api_key", "q"), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertIn("coldiq", main.provider_chain())

    def test_no_key_removes_it_rather_than_breaking_the_chain(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "crustdata_api_key", "c"), \
             patch.object(main.settings, "coldiq_api_key", None), \
             patch.object(main.settings, "search_provider", "bytemine"):
            chain = main.provider_chain()
            self.assertNotIn("coldiq", chain)
            self.assertIn("crustdata", chain)


class RoutedResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return json.loads(self.text)


class RoutedClient:
    """Fake httpx client that answers per-path and records what it was sent.

    ColdIQ's verbs all share one envelope, so the interesting assertions are
    which path was called, in what order, and with what body.
    """
    routes = {}
    calls = []
    raise_on = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        path = url.replace(main.COLDIQ_BASE, "")
        self.__class__.calls.append((path, json))
        if self.__class__.raise_on and self.__class__.raise_on in path:
            raise main.httpx.ConnectError("boom")
        status, payload = self.__class__.routes.get(path, (404, {"error": "no route"}))
        return RoutedResponse(status, payload)

    @classmethod
    def reset(cls, routes=None, raise_on=None):
        cls.routes = routes or {}
        cls.calls = []
        cls.raise_on = raise_on


class CredentialStrippingTests(unittest.TestCase):
    def test_it_covers_a_field_nobody_remembered_to_list(self):
        # The original validator named its fields, with a comment saying the
        # next provider could not forget it. The next provider forgot it:
        # getleads_api_key was added with the same trailing space and failed on
        # every call. "*" is the only version that cannot be forgotten.
        for field in ("getleads_api_key", "coldiq_api_key", "wiza_api_key",
                      "crustdata_api_key", "bytemine_api_key",
                      "anthropic_api_key", "database_url", "search_provider"):
            self.assertIn(field, main.Settings.model_fields, field)

        cleaned = main.Settings._strip_credential(
            "  glb_live_01M1CMVMEXGWBEDGG8VJN9TCZQ \n")
        self.assertEqual(cleaned, "glb_live_01M1CMVMEXGWBEDGG8VJN9TCZQ")

    def test_a_non_string_setting_is_returned_untouched(self):
        # "*" sees every field, including the ints and bools.
        self.assertEqual(main.Settings._strip_credential(3600), 3600)
        self.assertIs(main.Settings._strip_credential(True), True)


    """A pasted key keeps whatever whitespace came with it.

    httpx refuses to build a header whose value has trailing whitespace, so the
    provider fails on every single call — while the key still reads as
    configured and the provider stays in the chain looking healthy. ColdIQ ran
    that way from the day it was added: no leads, and every email verification
    answering "unknown".
    """

    def test_whitespace_is_stripped_from_every_credential(self):
        for field in ("wiza_api_key", "coldiq_api_key", "crustdata_api_key",
                      "bytemine_api_key", "anthropic_api_key"):
            cleaned = main.Settings._strip_credential(f"  secret_{field} \n")
            self.assertEqual(cleaned, f"secret_{field}", field)

    def test_a_stripped_key_makes_a_legal_header_value(self):
        # The exact failure: LocalProtocolError, Illegal header value.
        key = main.Settings._strip_credential("ciq_live_f2a8d598 ")
        header = f"Bearer {key}"

        self.assertEqual(header, "Bearer ciq_live_f2a8d598")
        self.assertEqual(header, header.strip())

    def test_none_and_non_strings_pass_through_untouched(self):
        self.assertIsNone(main.Settings._strip_credential(None))
        self.assertEqual(main.Settings._strip_credential(123), 123)


class ColdiqVerdictTests(unittest.TestCase):
    """The verify result is untyped in ColdIQ's spec, so the reader is tolerant."""

    def test_the_four_documented_states_map_to_themselves(self):
        for state in ("deliverable", "risky", "undeliverable", "unknown"):
            self.assertEqual(main.read_coldiq_verdict({"status": state})["status"], state)

    def test_vendor_spellings_are_folded_onto_our_vocabulary(self):
        for raw, expected in (("valid", "deliverable"), ("VERIFIED", "deliverable"),
                              ("invalid", "undeliverable"), ("bounced", "undeliverable"),
                              ("catch-all", "catch_all"), ("accept_all", "catch_all"),
                              ("disposable", "risky")):
            self.assertEqual(main.read_coldiq_verdict({"state": raw})["status"],
                             expected, raw)

    def test_an_unrecognised_verdict_is_unknown_not_a_pass(self):
        # Guessing "sendable" from a word we do not know is how a domain gets
        # burned; unknown leaves the decision to the caller.
        v = main.read_coldiq_verdict({"status": "wibble"})

        self.assertEqual(v["status"], "unknown")
        self.assertIsNone(v["sendable"])

    def test_flags_stand_in_for_a_missing_verdict(self):
        self.assertEqual(main.read_coldiq_verdict({"is_catch_all": True})["status"],
                         "catch_all")
        self.assertEqual(main.read_coldiq_verdict({"disposable": True})["status"], "risky")

    def test_catch_all_and_risky_are_sendable_but_undeliverable_is_not(self):
        # ColdIQ's own bulk docs call catch_all and risky conclusive and
        # sendable. Treating them as failures would drop most B2B addresses.
        self.assertTrue(main.read_coldiq_verdict({"status": "catch_all"})["sendable"])
        self.assertTrue(main.read_coldiq_verdict({"status": "risky"})["sendable"])
        self.assertFalse(main.read_coldiq_verdict({"status": "invalid"})["sendable"])

    def test_the_answering_vendor_is_carried_through(self):
        v = main.read_coldiq_verdict(
            {"status": "deliverable", "_meta": {"provider": "bounceban"}})

        self.assertEqual(v["vendor"], "bounceban")
        self.assertEqual(v["checked_by"], "coldiq")


class ColdiqVerifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_verdict_is_fetched_and_normalized(self):
        RoutedClient.reset({"/v1/email/verify": (200, {
            "data": {"status": "deliverable"}, "_meta": {"provider": "bounceban"}})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            v = await main.coldiq_verify_email("ada@acme.com")

        self.assertEqual(RoutedClient.calls[0][0], "/v1/email/verify")
        self.assertEqual(RoutedClient.calls[0][1]["input"], {"email": "ada@acme.com"})
        self.assertEqual(v["status"], "deliverable")
        self.assertTrue(v["sendable"])
        self.assertIn("checked_at", v)

    async def test_an_undeliverable_address_is_reported_not_hidden(self):
        RoutedClient.reset({"/v1/email/verify": (200, {"data": {"status": "invalid"}})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            v = await main.coldiq_verify_email("nobody@acme.com")

        self.assertEqual(v["status"], "undeliverable")
        self.assertFalse(v["sendable"])

    async def test_no_email_is_unverified_and_costs_nothing(self):
        RoutedClient.reset({})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            v = await main.coldiq_verify_email("")

        self.assertEqual(v["status"], "unverified")
        self.assertEqual(RoutedClient.calls, [])

    async def test_without_a_key_nothing_is_called(self):
        RoutedClient.reset({})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", None):
            v = await main.coldiq_verify_email("ada@acme.com")

        self.assertEqual(v["status"], "unverified")
        self.assertEqual(RoutedClient.calls, [])

    async def test_a_dead_verifier_never_costs_the_caller_the_reveal(self):
        # The reveal has already been paid for. A verification outage must
        # downgrade the verdict, not turn a successful enrich into an error.
        RoutedClient.reset({}, raise_on="/v1/email/verify")
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            v = await main.coldiq_verify_email("ada@acme.com")

        self.assertEqual(v["status"], "unknown")
        self.assertIsNone(v["sendable"])

    async def test_a_404_is_a_miss_rather_than_a_failure(self):
        RoutedClient.reset({"/v1/email/verify": (404, {"error": "no usable result"})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            v = await main.coldiq_verify_email("ada@acme.com")

        self.assertEqual(v["status"], "unknown")

    async def test_every_providers_lead_gets_the_same_check(self):
        RoutedClient.reset({"/v1/email/verify": (200, {"data": {"status": "catch_all"}})})
        for provider in ("bytemine", "crustdata", "coldiq", "wiza"):
            with patch.object(main.httpx, "AsyncClient", RoutedClient), \
                 patch.object(main.settings, "coldiq_api_key", "key"):
                lead = await main.verify_revealed_lead(
                    {"provider": provider, "business_email": "ada@acme.com"})

            self.assertEqual(lead["email_verification"]["status"], "catch_all", provider)
            self.assertTrue(lead["email_verified"], provider)

    async def test_a_lead_with_no_email_still_comes_back_whole(self):
        RoutedClient.reset({})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            lead = await main.verify_revealed_lead(
                {"provider": "wiza", "contact_name": "Ada"})

        self.assertEqual(lead["contact_name"], "Ada")
        self.assertEqual(lead["email_verification"]["status"], "unverified")
        self.assertIsNone(lead["email_verified"])


class ColdiqRevealTests(unittest.IsolatedAsyncioTestCase):
    def _request(self, **kwargs):
        return main.EnrichRequest(**kwargs)

    async def test_the_finder_is_tried_first_because_a_miss_is_free(self):
        RoutedClient.reset({"/v1/email/find": (200, {"data": {
            "email": "ada@acme.com", "first_name": "Ada", "company_name": "Acme"}})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            lead = await main.coldiq_reveal(
                self._request(linkedin_url="https://linkedin.com/in/ada"))

        self.assertEqual([c[0] for c in RoutedClient.calls], ["/v1/email/find"])
        self.assertEqual(lead["business_email"], "ada@acme.com")
        self.assertEqual(lead["provider"], "coldiq")

    async def test_a_finder_miss_falls_back_to_the_profile_enrich(self):
        RoutedClient.reset({
            "/v1/email/find": (404, {"error": "no usable result"}),
            "/v1/person/enrich": (200, {"data": {
                "full_name": "Ada Lovelace", "title": "CTO", "company_name": "Acme"}}),
        })
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            lead = await main.coldiq_reveal(
                self._request(linkedin_url="https://linkedin.com/in/ada"))

        self.assertEqual([c[0] for c in RoutedClient.calls],
                         ["/v1/email/find", "/v1/person/enrich"])
        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["job_title"], "CTO")

    async def test_a_name_without_a_company_is_not_sent_to_the_finder(self):
        # PersonIdentity needs a LinkedIn URL or a name paired with a company;
        # a bare name would be charged for a lookup that cannot resolve.
        RoutedClient.reset({"/v1/person/enrich": (404, {})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            await main.coldiq_reveal(self._request(full_name="Ada Lovelace"))

        self.assertNotIn("/v1/email/find", [c[0] for c in RoutedClient.calls])

    async def test_a_name_and_domain_is_enough_to_look_up(self):
        RoutedClient.reset({"/v1/email/find": (200, {"data": {"email": "ada@acme.com"}})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            await main.coldiq_reveal(
                self._request(full_name="Ada Lovelace", company_domain="acme.com"))

        sent = RoutedClient.calls[0][1]["input"]
        self.assertEqual(sent["first_name"], "Ada")
        self.assertEqual(sent["last_name"], "Lovelace")
        self.assertEqual(sent["domain"], "acme.com")

    async def test_nothing_found_returns_nothing_rather_than_an_empty_lead(self):
        RoutedClient.reset({"/v1/email/find": (404, {}), "/v1/person/enrich": (404, {})})
        with patch.object(main.httpx, "AsyncClient", RoutedClient), \
             patch.object(main.settings, "coldiq_api_key", "key"):
            lead = await main.coldiq_reveal(
                self._request(linkedin_url="https://linkedin.com/in/ada"))

        self.assertEqual(lead, {})


class PhoneTests(unittest.TestCase):
    """Line type is the closest thing to a WhatsApp signal the providers sell.

    None of them expose a WhatsApp flag, so "mobile" is what can honestly be
    reported. That makes it worth getting right: a wrongly-labelled office line
    is worse than an unlabelled number.
    """

    def test_each_providers_key_names_carry_the_line_type(self):
        # Wiza splits mobile_phone from phone_number, Bytemine mobile_phone
        # from direct_dial. Reading whichever came first threw that away.
        self.assertEqual(main.pick_phone({"mobile_phone": "+3247"}), ("+3247", "mobile"))
        self.assertEqual(main.pick_phone({"phone_number": "+3229"}), ("+3229", "office"))
        self.assertEqual(main.pick_phone({"direct_dial": "+3229"}), ("+3229", "office"))

    def test_a_mobile_wins_when_a_contact_has_both(self):
        number, kind = main.pick_phone(
            {"direct_dial": "+3229990000", "mobile_phone": "+32470123456"})

        self.assertEqual((number, kind), ("+32470123456", "mobile"))

    def test_a_bare_number_is_untyped_rather_than_assumed_mobile(self):
        # Claiming "mobile" for a number nobody typed would put a WhatsApp
        # badge on a switchboard.
        self.assertEqual(main.pick_phone({"phone": "+3229990000"}),
                         ("+3229990000", None))

    def test_an_array_entrys_own_type_is_read(self):
        contact = {"phones": [
            {"number": "+3229990000", "type": "work"},
            {"pretty_number": "+32 470 12 34 56", "type": "mobile"},
        ]}

        self.assertEqual(main.pick_phone(contact), ("+32 470 12 34 56", "mobile"))

    def test_crustdata_spells_the_array_differently(self):
        contact = {"phone_numbers": [{"number": "+12125551234", "type": "cell"}]}

        self.assertEqual(main.pick_phone(contact), ("+12125551234", "mobile"))

    def test_a_masked_number_is_no_number(self):
        # Bytemine masks withheld numbers with asterisks. Storing those as a
        # phone number is worse than storing nothing.
        self.assertEqual(main.pick_phone({"mobile_phone": "+3247*****"}), (None, None))
        self.assertEqual(main.pick_phone({"phones": [{"number": "***"}]}), (None, None))

    def test_nothing_at_all(self):
        self.assertEqual(main.pick_phone({}), (None, None))
        self.assertEqual(main.pick_phone(None), (None, None))

    def test_an_unknown_line_type_word_is_not_guessed_at(self):
        self.assertIsNone(main.classify_phone_type("satellite"))
        self.assertIsNone(main.classify_phone_type(""))
        self.assertEqual(main.classify_phone_type("Cellular"), "mobile")
        self.assertEqual(main.classify_phone_type("LANDLINE"), "office")

    def test_the_transforms_all_report_a_type(self):
        wiza = main.transform_wiza_contact({"full_name": "Ada", "mobile_phone": "+3247"})
        reveal = main.transform_reveal_contact({"name": "Ada", "phone_number": "+3229"})
        coldiq = main.transform_coldiq_profile({"first_name": "Ada", "mobile_phone": "+3247"})

        self.assertEqual((wiza["phone"], wiza["phone_type"]), ("+3247", "mobile"))
        self.assertEqual((reveal["phone"], reveal["phone_type"]), ("+3229", "office"))
        self.assertEqual((coldiq["phone"], coldiq["phone_type"]), ("+3247", "mobile"))


class RegionLocationTests(unittest.TestCase):
    """"Europe" is not a city, and sending it as one matches nothing.

    From production: `Bytemine /b2b-search body: {… "city": "Europe" …}`
    returning `total_companies: 0`, on every European search, while reporting
    success. A region has to be recognised as a region so it can be refused.
    """

    def test_continents_and_blocs_are_regions(self):
        for token in ("Europe", "EMEA", "APAC", "Asia", "North America",
                      "LATAM", "Nordics", "Middle East", "DACH"):
            kind, value = main.classify_location(token)
            self.assertEqual(kind, "region", token)
            self.assertEqual(value, token)

    def test_real_places_are_still_read_as_before(self):
        self.assertEqual(main.classify_location("California"), ("state", "CA"))
        self.assertEqual(main.classify_location("Germany"), ("country", "DE"))
        self.assertEqual(main.classify_location("San Francisco"),
                         ("city", "San Francisco"))
        self.assertEqual(main.classify_location("TX"), ("state", "TX"))

    def test_no_provider_pretends_a_region_is_a_place(self):
        # Each refuses rather than filtering on something that cannot match, so
        # the fan-out reaches Crustdata — which does match a region by name.
        for build in (main.build_bytemine_filters, main.build_coldiq_filters,
                      main.build_wiza_filters):
            with self.assertRaises(main.ProviderUnsupported, msg=build.__name__):
                build({"job_title": "Founder", "location": "Europe"})

    def test_the_company_keyword_search_refuses_a_region_too(self):
        # This is the one that was sending city:"Europe" — it has a country
        # field, so it had somewhere to put a country and nowhere to put a
        # continent, and the city branch swallowed it.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_bytemine_company_body({"keywords": "AI", "location": "Europe"})

    def test_the_company_search_still_places_what_it_can(self):
        by_country = main.build_bytemine_company_body(
            {"keywords": "AI", "location": "Germany"})
        by_city = main.build_bytemine_company_body(
            {"keywords": "AI", "location": "Berlin"})

        self.assertEqual(by_country["country"], "DE")
        self.assertNotIn("city", by_country)
        self.assertEqual(by_city["city"], "Berlin")

    def test_crustdata_still_takes_a_region_because_it_can(self):
        body = main.build_crustdata_filters(
            {"job_title": "Founder", "location": "Europe"})

        rendered = json.dumps(body)
        self.assertIn("Europe", rendered)
        self.assertIn("full_location", rendered)


class SeniorityImpliedByTitleTests(unittest.TestCase):
    """One stated criterion must not become two ANDed filters.

    "AI SaaS founders" parses to job_title="Founder" AND seniority="founder" —
    one word, two filters. Sending both asserts our seniority taxonomy on top of
    the title, and when the provider disagrees the second filter removes
    everyone the first one found. In production Bytemine and Crustdata both
    returned zero for `title ~ Founder AND seniority = Owner`, while Wiza
    returned real people for the identical search.
    """

    def test_a_title_that_states_the_seniority_implies_it(self):
        for title, seniority in (("Founder", "founder"), ("Founder", "owner"),
                                 ("Co-Founder", "founder"), ("CEO", "cxo"),
                                 ("CEO", "c_suite"), ("VP of Sales", "vp"),
                                 ("Head of Growth", "manager"),
                                 ("Director of Ops", "director")):
            self.assertTrue(main.seniority_implied_by_title(title, seniority),
                            f"{title} / {seniority}")

    def test_a_seniority_the_title_does_not_state_is_a_real_criterion(self):
        for title, seniority in (("Account Executive", "vp"),
                                 ("Engineer", "founder"),
                                 ("Sales Manager", "vp"),
                                 ("Founder", "vp")):
            self.assertFalse(main.seniority_implied_by_title(title, seniority),
                             f"{title} / {seniority}")

    def test_spelling_differences_are_not_two_criteria(self):
        # "CEO" + "cxo" is one thing said twice; comparing raw strings missed it.
        self.assertTrue(main.seniority_implied_by_title("CEO", "c-level"))
        self.assertTrue(main.seniority_implied_by_title("VP Sales", "vice president"))

    def test_nothing_is_implied_by_a_missing_side(self):
        self.assertFalse(main.seniority_implied_by_title("", "founder"))
        self.assertFalse(main.seniority_implied_by_title("Founder", ""))
        self.assertFalse(main.seniority_implied_by_title(None, None))

    def test_the_redundant_filter_is_dropped_for_every_provider(self):
        params = {"job_title": "Founder", "seniority": "founder"}

        self.assertNotIn("seniorityLevels", main.build_bytemine_filters(params))
        self.assertNotIn("seniorities", main.build_coldiq_filters(params))
        self.assertNotIn("job_title_level", main.build_wiza_filters(params))
        self.assertNotIn("seniority_level",
                         json.dumps(main.build_crustdata_filters(params)))

    def test_the_title_the_user_said_still_constrains_the_search(self):
        # Dropping the implied seniority must not widen the search past what
        # was asked for: the word is still in the request.
        body = main.build_bytemine_filters(
            {"job_title": "Founder", "seniority": "founder"})

        self.assertEqual(body["jobTitles"], ["Founder"])

    def test_a_genuinely_separate_seniority_is_still_sent(self):
        params = {"job_title": "Account Executive", "seniority": "vp"}

        self.assertEqual(main.build_bytemine_filters(params)["seniorityLevels"], ["VP"])
        self.assertEqual(main.build_wiza_filters(params)["job_title_level"], ["VP"])


class EnrichPlaceholderTests(unittest.TestCase):
    """A placeholder must never become a search filter.

    Production sent Wiza {"full_name": "Ott Salmar", "company": "Unknown"} — a
    real person scoped to a company literally named Unknown, so the reveal could
    only miss. The string is written upstream because leads.company_name is NOT
    NULL and the agent needs something when a provider returns no company; it is
    a reasonable column value and never an identifier.
    """

    def test_the_reported_case(self):
        request = main.EnrichRequest(full_name="Ott Salmar", company="Unknown")

        self.assertIsNone(request.company)
        self.assertEqual(request.full_name, "Ott Salmar")

    def test_every_spelling_of_we_do_not_know(self):
        for value in ("Unknown", "unknown", "N/A", "n/a", "na", "none", "NULL",
                      "-", "--", "?", "TBD", "not available", "undefined", "  "):
            request = main.EnrichRequest(full_name="Ada", company=value)
            self.assertIsNone(request.company, value)

    def test_a_real_name_that_merely_contains_one_is_kept(self):
        # Anchored matching. These are real companies, not placeholders.
        for value in ("NA Consulting", "Nan Systems", "Company House Ltd",
                      "None The Wiser Ltd", "TBD Partners"):
            request = main.EnrichRequest(full_name="Ada", company=value)
            self.assertEqual(request.company, value, value)

    def test_it_applies_to_every_identifier_not_just_company(self):
        request = main.EnrichRequest(
            full_name="unknown", company="Acme", company_domain="N/A",
            email="none", linkedin_url="  ")

        self.assertIsNone(request.full_name)
        self.assertIsNone(request.company_domain)
        self.assertIsNone(request.email)
        self.assertIsNone(request.linkedin_url)
        self.assertEqual(request.company, "Acme")

    def test_surrounding_whitespace_is_trimmed_from_real_values(self):
        request = main.EnrichRequest(full_name=" Ada Lovelace ",
                                     email=" ada@acme.com ")

        self.assertEqual(request.full_name, "Ada Lovelace")
        self.assertEqual(request.email, "ada@acme.com")


class ColdiqHeadlineRecordTests(unittest.TestCase):
    """The waterfall can answer a people search with LinkedIn search snippets.

    Confirmed from production once the key was fixed and ColdIQ ran for the
    first time:

        ColdIQ record keys: ['_unverified_dimensions', 'linkedin_url', 'title']
        {"title": "Seb Hall - Founder @ Cloud Employee | Helping US & UK ...",
         "linkedin_url": "https://www.linkedin.com/in/seb-hall"}

    Two fields, with the name glued to the front of the title. Read literally
    that is a lead with no name and no company, which the frontend drops — so
    the log said "coldiq returned 6 leads" and the user saw an empty page.

    Every string below is a real one from that log.
    """

    def test_the_records_that_showed_the_user_nothing(self):
        for headline, expected in (
            ("Seb Hall - Founder @ Cloud Employee | Helping US & UK ...",
             ("Seb Hall", "Founder", "Cloud Employee")),
            ("Mark Lucovsky - Founder",
             ("Mark Lucovsky", "Founder", None)),
            ("Jim Whitson - Owner, WitWerx, Inc.",
             ("Jim Whitson", "Owner", "WitWerx, Inc.")),
            ("Kevin Steward - Software Engineer at Google",
             ("Kevin Steward", "Software Engineer", "Google")),
            ("Vance Wood - Google Engineer | Founder",
             ("Vance Wood", "Google Engineer", None)),
        ):
            self.assertEqual(
                main.read_headline_record({"title": headline}), expected, headline)

    def test_positioning_copy_after_a_pipe_is_not_a_job_title(self):
        _, title, _ = main.read_headline_record(
            {"title": "Seb Hall - Founder @ Cloud Employee | Helping US & UK ..."})

        self.assertEqual(title, "Founder")

    def test_a_company_is_only_taken_when_the_headline_names_one(self):
        # "@" and "at" are explicit. A bare comma is not: "Engineer, Senior" is
        # not a person at a company called Senior.
        _, title, company = main.read_headline_record(
            {"title": "Dana Scully - Engineer, Senior"})

        self.assertEqual(title, "Engineer")
        self.assertIsNone(company)

    def test_a_corporate_suffix_does_mark_a_company(self):
        _, _, company = main.read_headline_record(
            {"title": "Jim Whitson - Owner, WitWerx, Inc."})

        self.assertEqual(company, "WitWerx, Inc.")

    def test_the_linkedin_slug_is_the_fallback_name(self):
        self.assertEqual(
            main.name_from_linkedin_url("https://www.linkedin.com/in/seb-hall"),
            "Seb Hall")
        # LinkedIn's disambiguating id is not part of anyone's name.
        self.assertEqual(
            main.name_from_linkedin_url("https://www.linkedin.com/in/mark-lucovsky-5280034"),
            "Mark Lucovsky")
        self.assertEqual(
            main.name_from_linkedin_url("https://www.linkedin.com/in/kevin-steward-b91628117"),
            "Kevin Steward")

    def test_an_unsplittable_slug_is_left_alone_rather_than_mangled(self):
        self.assertIsNone(main.name_from_linkedin_url("https://linkedin.com/in/juliendanjou"))
        self.assertIsNone(main.name_from_linkedin_url(""))
        self.assertIsNone(main.name_from_linkedin_url("https://example.com/nobody"))

    def test_a_headline_with_a_dash_is_not_read_as_a_four_word_name(self):
        name, _, _ = main.read_headline_record(
            {"title": "Building the future of sales - Founder"})

        self.assertIsNone(name)

    def test_the_thin_record_becomes_a_lead_the_frontend_will_show(self):
        lead = main.transform_coldiq_profile({
            "title": "Seb Hall - Founder @ Cloud Employee | Helping US & UK ...",
            "linkedin_url": "https://www.linkedin.com/in/seb-hall",
        })

        self.assertEqual(lead["contact_name"], "Seb Hall")
        self.assertEqual(lead["job_title"], "Founder")
        self.assertEqual(lead["company_name"], "Cloud Employee")
        # The frontend filters on exactly this.
        self.assertTrue(lead["contact_name"] or lead["company_name"])

    def test_a_full_record_is_untouched_by_any_of_this(self):
        lead = main.transform_coldiq_profile({
            "first_name": "Ada", "last_name": "Lovelace",
            "company_name": "Acme", "title": "CTO", "email": "ada@acme.com",
        })

        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["job_title"], "CTO")
        self.assertEqual(lead["company_name"], "Acme")
        self.assertEqual(lead["business_email"], "ada@acme.com")


class LinkedInIdentityTests(unittest.TestCase):
    """One definition of "the same person", because several places ask.

    ColdIQ's search returns https://www.linkedin.com/in/seb-hall and its enrich
    returns http://www.linkedin.com/in/seb-hall for that same person, so a raw
    string comparison treats them as two people and shows him twice.
    """

    def test_every_spelling_of_one_profile_is_one_identity(self):
        spellings = [
            "https://www.linkedin.com/in/seb-hall",
            "http://www.linkedin.com/in/seb-hall",
            "https://linkedin.com/in/seb-hall",
            "https://www.linkedin.com/in/seb-hall/",
            "HTTPS://WWW.LINKEDIN.COM/in/seb-hall",
        ]

        self.assertEqual(len({main.linkedin_identity(u) for u in spellings}), 1)

    def test_different_people_stay_different(self):
        self.assertNotEqual(
            main.linkedin_identity("https://www.linkedin.com/in/seb-hall"),
            main.linkedin_identity("https://www.linkedin.com/in/jim-whitson"))

    def test_anything_that_is_not_a_profile_is_no_identity(self):
        for value in ("", None, "https://example.com/in/seb-hall", "not a url"):
            self.assertIsNone(main.linkedin_identity(value), repr(value))

    def test_a_coldiq_record_is_trackable(self):
        # _profile_url read only Crustdata's nested shape, so every ColdIQ
        # record was untrackable: campaign history recorded none of them and
        # the same people came back on every search.
        record = {"linkedin_url": "https://www.linkedin.com/in/seb-hall",
                  "title": "Seb Hall - Founder @ Cloud Employee"}

        self.assertEqual(main._profile_url(record),
                         "https://www.linkedin.com/in/seb-hall")
        self.assertIsNotNone(main._profile_lead_key(record))

    def test_the_crustdata_shape_still_wins_where_it_exists(self):
        record = {
            "social_handles": {"professional_network_identifier": {
                "profile_url": "https://www.linkedin.com/in/nested"}},
            "linkedin_url": "https://www.linkedin.com/in/flat",
        }

        self.assertEqual(main._profile_url(record),
                         "https://www.linkedin.com/in/nested")

    def test_a_lead_dedupes_on_the_normalised_identity(self):
        a = main._dedupe_key({"linkedin_url": "https://www.linkedin.com/in/seb-hall"})
        b = main._dedupe_key({"linkedin_url": "http://linkedin.com/in/seb-hall/"})

        self.assertEqual(a, b)


class GetleadsFilterTests(unittest.TestCase):
    """The first provider in the chain that can express a whole ICP.

    A search for "founders at 1-10 employee AI SaaS" returned a Software
    Engineer at Google, because the only provider still answering was one with
    no industry field and no headcount field. These are hard filters.
    """

    def test_a_whole_icp_becomes_hard_filters(self):
        f = main.build_getleads_filters({
            "job_title": "Founder",
            "industry": "computer software",
            "company_size": "1-10",
            "location": "United States",
        })

        self.assertEqual(f["job_titles"], ["Founder"])
        self.assertEqual(f["industries"], ["Computer Software"])
        self.assertEqual(f["employees_min"], 1)
        self.assertEqual(f["employees_max"], 10)
        self.assertEqual(f["countries"], ["United States"])

    def test_a_region_is_answerable_here_rather_than_refused(self):
        # Every other provider steps aside on a continent. This one has fields
        # for both a macro-region and a continent, so "Europe" is a real filter.
        self.assertEqual(
            main.build_getleads_filters({"job_title": "Owner", "location": "Europe"})["continents"],
            ["Europe"])
        self.assertEqual(
            main.build_getleads_filters({"job_title": "Owner", "location": "EMEA"})["regions"],
            ["EMEA"])
        self.assertEqual(
            main.build_getleads_filters({"job_title": "Owner", "location": "APAC"})["regions"],
            ["APAC"])

    def test_a_country_code_is_turned_back_into_the_name_the_filter_wants(self):
        # classify_location yields ISO-2; this filter matches on country names.
        for token, expected in (("US", "United States"), ("Germany", "Germany"),
                                ("UK", "United Kingdom")):
            f = main.build_getleads_filters({"job_title": "CTO", "location": token})
            self.assertEqual(f["countries"], [expected], token)

    def test_a_state_arrives_as_its_name(self):
        f = main.build_getleads_filters({"job_title": "CTO", "location": "California"})

        self.assertEqual(f["states"], ["California"])

    def test_an_unmappable_industry_is_refused_rather_than_guessed(self):
        with self.assertRaises(main.ProviderUnsupported):
            main.build_getleads_filters(
                {"job_title": "Founder", "industry": "underwater basket weaving"})

    def test_the_implied_seniority_is_dropped_here_too(self):
        implied = main.build_getleads_filters(
            {"job_title": "Founder", "seniority": "founder"})
        separate = main.build_getleads_filters(
            {"job_title": "Account Executive", "seniority": "vp"})

        self.assertNotIn("seniority", implied)
        self.assertEqual(separate["seniority"], ["VP"])

    def test_a_segment_phrase_searches_the_company_description(self):
        f = main.build_getleads_filters(
            {"job_title": "Founder", "keywords": "AI SaaS"})

        self.assertEqual(f["company_description"], "AI SaaS")

    def test_a_domain_beats_a_company_name(self):
        by_domain = main.build_getleads_filters({"company_domain": "acme.com"})
        by_name = main.build_getleads_filters({"company": "Acme Inc"})
        # The parser emitting a bare domain as a company name is read as one.
        as_domain = main.build_getleads_filters({"company": "workflows.io"})

        self.assertEqual(by_domain["domains"], ["acme.com"])
        self.assertEqual(by_name["company_name"], "Acme Inc")
        self.assertEqual(as_domain["domains"], ["workflows.io"])


class GetleadsTransformTests(unittest.TestCase):
    def test_the_documented_row_shape(self):
        lead = main.transform_getleads_contact({
            "first_name": "Jane", "last_name": "Doe",
            "email_address": "jane@acme.com", "email_status": "VALID",
            "cellphone": "+1 415-555-0142",
            "domain_org": "acme.com", "org_company_name": "Acme Inc",
            "person_country_name": "United States",
            "person_linkedin_url": "https://www.linkedin.com/in/janedoe",
            "title": "VP Sales",
        })

        self.assertEqual(lead["contact_name"], "Jane Doe")
        self.assertEqual(lead["business_email"], "jane@acme.com")
        self.assertEqual(lead["email_status"], "VALID")
        self.assertEqual(lead["company_name"], "Acme Inc")
        self.assertEqual(lead["company_domain"], "acme.com")
        self.assertEqual(lead["job_title"], "VP Sales")
        self.assertEqual(lead["country"], "United States")
        self.assertEqual(lead["provider"], "getleads")
        # cellphone is a mobile by name, which is the WhatsApp-capable one.
        self.assertEqual((lead["phone"], lead["phone_type"]),
                         ("+1 415-555-0142", "mobile"))

    def test_the_camelcase_convenience_keys_are_accepted_too(self):
        lead = main.transform_getleads_contact({
            "firstName": "Ada", "lastName": "Lovelace",
            "emailAddress": "ada@acme.com", "linkedinUrl": "https://x.com/a",
        })

        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["business_email"], "ada@acme.com")

    def test_an_empty_row_yields_nulls_rather_than_junk(self):
        lead = main.transform_getleads_contact({})

        self.assertIsNone(lead["contact_name"])
        self.assertIsNone(lead["business_email"])
        self.assertFalse(lead["phone_available"])

    def test_a_getleads_row_is_trackable_so_it_is_not_shown_twice(self):
        row = {"person_linkedin_url": "https://www.linkedin.com/in/janedoe"}

        self.assertIsNotNone(main._profile_lead_key(row))
        self.assertEqual(main.linkedin_identity(main._profile_url(row)), "in/janedoe")


class GetleadsChainTests(unittest.TestCase):
    def test_it_joins_the_chain_ahead_of_coldiq(self):
        order = list(main.PROVIDER_ORDER)

        self.assertIn("getleads", order)
        self.assertLess(order.index("getleads"), order.index("coldiq"))

    def test_no_key_removes_it_rather_than_breaking_the_chain(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "getleads_api_key", None), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertNotIn("getleads", main.provider_chain())

    def test_a_key_is_all_it_takes(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "getleads_api_key", "glb_live_x"), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertIn("getleads", main.provider_chain())


class GetleadsRepeatLeadTests(unittest.IsolatedAsyncioTestCase):
    """Every search must surface people the user has not been shown before.

    start_offset is zero on every fresh search — the frontend sends page 1 each
    time — so this leg fetched the same first rows for the same ICP and then
    removed the ones already seen. Re-run the same ICP and that page is entirely
    seen: the provider still answers and still bills, and the user gets nothing
    new. It is the one leg with a real offset, so it can go and look further in.
    """

    def _pages(self, pages):
        """Stub getleads_person_search: a dict of offset -> rows."""
        seen_offsets: list = []

        async def search(params, limit, offset=0):
            seen_offsets.append(offset)
            rows = pages.get(offset, [])
            nxt = offset + limit
            return {"profiles": rows, "total": sum(len(p) for p in pages.values()),
                    "next_offset": nxt if nxt in pages else None}

        return search, seen_offsets

    async def run_search(self, request, pages):
        search, offsets = self._pages(pages)

        async def no_cache(_hash):
            return None

        async def no_store(*args, **kwargs):
            return None

        patches = [
            patch.object(main.settings, "getleads_api_key", "glb_live_x"),
            patch.object(main, "provider_chain", lambda: ("getleads",)),
            patch.object(main, "getleads_person_search", search),
            patch.object(main, "cache_lookup", no_cache),
            patch.object(main, "cache_store", no_store),
        ]
        for p in patches:
            p.start()
        try:
            return await main.search_leads(request), offsets
        finally:
            for p in patches:
                p.stop()

    def _row(self, slug):
        return {"first_name": slug, "last_name": "X", "job_title": "Founder",
                "org_company_name": "Acme", "org_domain": "acme.com",
                "person_linkedin_url": f"https://www.linkedin.com/in/{slug}"}

    async def test_a_fully_seen_page_pages_forward_instead_of_returning_nothing(self):
        response, offsets = await self.run_search(
            main.SearchRequest(
                job_title="Founder", industry="computer software", limit=2,
                exclude_profiles=["https://www.linkedin.com/in/ada",
                                  "https://www.linkedin.com/in/bob"]),
            {0: [self._row("ada"), self._row("bob")],
             2: [self._row("cleo"), self._row("dev")]},
        )

        self.assertEqual(offsets, [0, 2])
        names = [lead["first_name"] for lead in response.leads]
        self.assertEqual(names, ["cleo", "dev"])

    async def test_a_page_of_new_people_costs_exactly_one_call(self):
        # Paging is for when there is nothing new, not something to do always:
        # this provider bills per record returned.
        response, offsets = await self.run_search(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               limit=2),
            {0: [self._row("ada"), self._row("bob")],
             2: [self._row("cleo")]},
        )

        self.assertEqual(offsets, [0])
        self.assertEqual([lead["first_name"] for lead in response.leads],
                         ["ada", "bob"])

    async def test_paging_is_bounded_rather_than_walking_the_index(self):
        seen = [f"https://www.linkedin.com/in/p{i}" for i in range(20)]
        pages = {i: [self._row(f"p{i}")] for i in range(0, 20)}

        _, offsets = await self.run_search(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               limit=1, exclude_profiles=seen),
            pages,
        )

        self.assertEqual(len(offsets), main.GETLEADS_MAX_PAGES)

    async def test_running_out_of_rows_stops_the_walk(self):
        response, offsets = await self.run_search(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               limit=2,
                               exclude_profiles=["https://www.linkedin.com/in/ada"]),
            {0: [self._row("ada")]},
        )

        self.assertEqual(offsets, [0])
        self.assertEqual(response.leads, [])


class RecoveredSegmentTermTests(unittest.IsolatedAsyncioTestCase):
    """The AI in "B2B AI SaaS founders" has to reach the provider answering.

    Only Crustdata reads semantic_query. When it returns nothing — which is most
    searches — the leg that actually answers saw job_title + industry and
    nothing else, so the user got any founder at any small software company
    anywhere and reasonably said the leads were not what they searched for.
    """

    async def resolve(self, **kwargs):
        return await main.resolve_search_params(main.SearchRequest(**kwargs))

    async def test_the_segment_is_recovered_from_the_sentence(self):
        params = await self.resolve(
            query="Founders at B2B AI SaaS companies with a team size of 1-10",
            job_title="Founder", industry="computer software", company_size="1-10")

        self.assertEqual(params["semantic_keywords"], "ai saas")

    async def test_it_is_not_promoted_to_a_stated_keyword(self):
        # keywords means "a criterion the user stated", and a provider with no
        # free-text field refuses the whole search rather than drop one. A term
        # lifted out of the sentence must not sideline Wiza.
        params = await self.resolve(
            query="Founders at AI SaaS companies", job_title="Founder",
            industry="computer software")

        self.assertNotIn("keywords", params)
        main.build_wiza_filters(params)  # does not raise

    async def test_a_stated_keyword_is_left_alone(self):
        params = await self.resolve(
            query="Founders at AI SaaS companies", job_title="Founder",
            keywords="devtools")

        self.assertEqual(params["keywords"], "devtools")
        self.assertNotIn("semantic_keywords", params)

    async def test_a_sentence_with_no_segment_term_adds_nothing(self):
        params = await self.resolve(
            query="heads of RevOps at fintechs", job_title="VP Sales",
            industry="financial services")

        self.assertNotIn("semantic_keywords", params)

    def test_getleads_matches_it_against_company_descriptions(self):
        f = main.build_getleads_filters(
            {"job_title": "Founder", "semantic_keywords": "ai saas"})

        self.assertEqual(f["company_description"], "ai saas")

    def test_a_stated_keyword_still_wins(self):
        f = main.build_getleads_filters(
            {"job_title": "Founder", "keywords": "devtools",
             "semantic_keywords": "ai saas"})

        self.assertEqual(f["company_description"], "devtools")


class FindymailClient:
    """Fake httpx client for Findymail: answers per (method, path)."""
    routes = {}
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, headers=None, json=None, params=None):
        path = url.replace(main.FINDYMAIL_BASE, "")
        self.__class__.calls.append((method, path, json, params))
        key = (method, path)
        status, payload = self.__class__.routes.get(key, (404, {}))
        if callable(payload):
            payload = payload(len([c for c in self.__class__.calls if c[1] == path]))
        return RoutedResponse(status, payload)

    @classmethod
    def reset(cls, routes=None):
        cls.routes = routes or {}
        cls.calls = []


def _fm(**env):
    """Patch settings for a Findymail test."""
    return patch.multiple(main.settings, findymail_api_key="fm_key", **env)


class FindymailSearchBodyTests(unittest.TestCase):
    """Intellimatch takes a sentence about companies, not a filter object."""

    def test_the_users_own_sentence_is_used_verbatim(self):
        # This is the one provider in the chain that wants the raw request.
        query = main.findymail_intellimatch_query(
            {"semantic_query": "Founders at B2B AI SaaS companies", "industry": "computer software"})

        self.assertEqual(query, "Founders at B2B AI SaaS companies")

    def test_structured_filters_are_written_back_out_as_a_sentence(self):
        query = main.findymail_intellimatch_query(
            {"industry": "computer software", "company_size": "1-10",
             "location": "United States"})

        self.assertEqual(
            query, "computer software companies with 1-10 employees in United States")

    def test_a_search_with_nothing_to_describe_companies_by_is_refused(self):
        # An empty query is rejected by Findymail, and a bare job title
        # describes a person rather than a company.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_findymail_search({"job_title": "Founder"}, 10)

    def test_require_email_is_on_so_a_miss_costs_nothing(self):
        body = main.build_findymail_search({"semantic_query": "AI SaaS"}, 6)

        self.assertTrue(body["config"]["require_email"])
        self.assertTrue(body["config"]["find_contact"])
        self.assertTrue(body["config"]["find_email"])

    def test_the_job_title_becomes_the_first_tier_not_an_alternative(self):
        body = main.build_findymail_search(
            {"semantic_query": "AI SaaS", "job_title": "Founder"}, 6)

        # Tiers are fallbacks: tier two is only tried when tier one finds
        # nobody, so a flat list would change the meaning.
        self.assertEqual(body["config"]["target_job_titles"], [["Founder"]])
        self.assertEqual(body["config"]["mode"], "targeted")

    def test_the_limit_is_clamped_to_their_documented_ceiling(self):
        self.assertEqual(main.build_findymail_search({"semantic_query": "x"}, 99999)["limit"], 5000)
        self.assertEqual(main.build_findymail_search({"semantic_query": "x"}, 0)["limit"], 1)


class FindymailTransformTests(unittest.TestCase):
    def test_the_documented_row_shape(self):
        lead = main.transform_findymail_row({
            "name": "Acme Corp", "domain": "acme.com",
            "employee_count_range": "51-200", "industries": ["Technology", "Software"],
            "country": "FR", "match_score": 95,
            "contact_name": "John Doe", "contact_email": "john.doe@acme.com",
            "contact_job_title": "CEO", "contact_phone": "+33 6 12 34 56 78",
            "contact_linkedin_url": "https://www.linkedin.com/in/johndoe",
        })

        self.assertEqual(lead["contact_name"], "John Doe")
        self.assertEqual(lead["first_name"], "John")
        self.assertEqual(lead["last_name"], "Doe")
        self.assertEqual(lead["business_email"], "john.doe@acme.com")
        self.assertEqual(lead["company_name"], "Acme Corp")
        self.assertEqual(lead["company_domain"], "acme.com")
        self.assertEqual(lead["job_title"], "CEO")
        self.assertEqual(lead["company_size"], "51-200")
        self.assertEqual(lead["industry"], "Technology")
        self.assertEqual(lead["provider"], "findymail")
        self.assertEqual(lead["phone"], "+33 6 12 34 56 78")

    def test_their_match_score_is_carried_not_adopted_as_ours(self):
        # It measures how well the company fits the query, which is not what
        # our fit score measures. One provider's number must not become ours.
        lead = main.transform_findymail_row({"name": "Acme", "match_score": 95})

        self.assertEqual(lead["match_score"], 95)
        self.assertNotIn("icp_score", lead)

    def test_an_empty_row_yields_nulls_rather_than_junk(self):
        lead = main.transform_findymail_row({})

        self.assertIsNone(lead["contact_name"])
        self.assertIsNone(lead["business_email"])
        self.assertFalse(lead["phone_available"])

    def test_a_findymail_row_is_trackable_so_it_is_not_shown_twice(self):
        row = {"contact_linkedin_url": "https://www.linkedin.com/in/johndoe"}

        self.assertEqual(main._profile_url(row),
                         "https://www.linkedin.com/in/johndoe")
        self.assertEqual(main.linkedin_identity(main._profile_url(row)), "in/johndoe")


class FindymailFindEmailTests(unittest.IsolatedAsyncioTestCase):
    """The reveal leg. Charged only on a found email, so a miss costs nothing."""

    async def test_a_linkedin_url_is_tried_first(self):
        # It identifies a person exactly; name plus domain can land on the
        # wrong person at a big company.
        FindymailClient.reset({
            ("POST", "/api/search/business-profile"):
                (200, {"contact": {"name": "Seb Hall", "domain": "cloudemployee.co.uk",
                                   "email": "seb.hall@cloudemployee.co.uk"}}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            contact = await main.findymail_find_email(
                linkedin_url="https://www.linkedin.com/in/seb-hall",
                name="Seb Hall", domain="cloudemployee.co.uk")

        self.assertEqual(contact["email"], "seb.hall@cloudemployee.co.uk")
        # Name search never ran — the first identifier answered.
        self.assertEqual([c[1] for c in FindymailClient.calls],
                         ["/api/search/business-profile"])

    async def test_it_falls_back_to_name_and_domain(self):
        FindymailClient.reset({
            ("POST", "/api/search/business-profile"): (200, {"contact": None}),
            ("POST", "/api/search/name"):
                (200, {"contact": {"name": "Ada Lovelace", "domain": "acme.com",
                                   "email": "ada@acme.com"}}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            contact = await main.findymail_find_email(
                linkedin_url="https://www.linkedin.com/in/ada",
                name="Ada Lovelace", domain="acme.com")

        self.assertEqual(contact["email"], "ada@acme.com")
        self.assertEqual([c[1] for c in FindymailClient.calls],
                         ["/api/search/business-profile", "/api/search/name"])

    async def test_a_miss_is_nothing_rather_than_an_error(self):
        FindymailClient.reset({
            ("POST", "/api/search/business-profile"): (200, {"contact": None}),
            ("POST", "/api/search/name"): (200, {"contact": None}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            contact = await main.findymail_find_email(
                linkedin_url="https://x/in/a", name="A B", domain="acme.com")

        self.assertEqual(contact, {})

    async def test_out_of_credits_does_not_raise(self):
        # 402 is a state of their account, not a failure of this request; a leg
        # that raises takes the whole reveal down with it.
        FindymailClient.reset({
            ("POST", "/api/search/business-profile"): (402, {"error": "Not enough credits"}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            contact = await main.findymail_find_email(
                linkedin_url="https://www.linkedin.com/in/seb-hall")

        self.assertEqual(contact, {})


class FindymailDomainTests(unittest.TestCase):
    """/api/search/name needs a mail domain, not a company name."""

    def test_a_company_name_is_not_a_domain(self):
        # Production sent domain="Uncapped" for every lead known only by name.
        self.assertIsNone(main.findymail_domain_for(None, "Uncapped"))
        self.assertIsNone(main.findymail_domain_for(None, "Cloud Employee"))

    def test_a_real_domain_is_used(self):
        self.assertEqual(main.findymail_domain_for("acme.com", "Acme"), "acme.com")

    def test_a_company_field_holding_a_domain_still_counts(self):
        # The UI puts a domain in either field depending on the provider.
        self.assertEqual(main.findymail_domain_for(None, "acme.com"), "acme.com")

    def test_a_url_is_reduced_to_its_host(self):
        self.assertEqual(
            main.findymail_domain_for("https://www.acme.com/about", None), "acme.com")

    def test_nothing_at_all_is_none(self):
        self.assertIsNone(main.findymail_domain_for(None, None))


class LinkedInIndustryTests(unittest.TestCase):
    """Bytemine and Crustdata both name industries the way LinkedIn does."""

    def test_our_lowercase_term_becomes_their_spelling(self):
        self.assertEqual(main.linkedin_industry("computer software"), "Computer Software")
        self.assertEqual(main.linkedin_industry("financial services"), "Financial Services")

    def test_their_spelling_survives_unchanged(self):
        self.assertEqual(main.linkedin_industry("Computer Software"), "Computer Software")

    def test_an_unknown_industry_is_none_rather_than_a_guess(self):
        self.assertIsNone(main.linkedin_industry("vertical ai agents"))

    def test_bytemine_still_refuses_what_it_cannot_map(self):
        # Crustdata passes an unmapped term through; Bytemine's is an enum, so
        # it steps aside instead of zeroing the search silently.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_bytemine_filters({"job_title": "Founder",
                                         "industry": "vertical ai agents"})

    def test_crustdata_passes_an_unmapped_industry_through(self):
        filters = main.build_crustdata_filters({"industry": "vertical ai agents"})
        self.assertEqual(filters["value"], "vertical ai agents")


class FindymailVerifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_verified_address_is_deliverable_and_sendable(self):
        FindymailClient.reset({
            ("POST", "/api/verify"):
                (200, {"email": "john@example.com", "verified": True, "provider": "Google"}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            verdict = await main.findymail_verify_email("john@example.com")

        self.assertEqual(verdict["status"], "deliverable")
        self.assertTrue(verdict["sendable"])
        self.assertEqual(verdict["checked_by"], "findymail")
        self.assertEqual(verdict["vendor"], "Google")

    async def test_an_unverified_address_is_undeliverable(self):
        FindymailClient.reset({
            ("POST", "/api/verify"): (200, {"email": "x@y.com", "verified": False}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            verdict = await main.findymail_verify_email("x@y.com")

        self.assertEqual(verdict["status"], "undeliverable")
        self.assertFalse(verdict["sendable"])

    async def test_nothing_is_invented_between_the_two(self):
        # Findymail returns a bare boolean, with none of ColdIQ's catch-all or
        # role detail. An error is unknown, never a guess either way.
        FindymailClient.reset({("POST", "/api/verify"): (402, {"error": "Not enough credits"})})
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            verdict = await main.findymail_verify_email("x@y.com")

        self.assertEqual(verdict["status"], "unknown")
        self.assertIsNone(verdict["sendable"])
        self.assertIn("credits", verdict["reason"])


class FindymailVerifierFallbackTests(unittest.IsolatedAsyncioTestCase):
    """ColdIQ is the primary checker and the one that runs out of credits.

    Production spent weeks answering "unknown" for every address because of it.
    """

    async def test_findymail_answers_when_coldiq_cannot(self):
        FindymailClient.reset({
            ("POST", "/api/verify"): (200, {"email": "a@b.com", "verified": True}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), \
             patch.object(main, "coldiq_verify_email", AsyncMock(return_value={
                 "status": "unknown", "sendable": None, "checked_by": "coldiq",
                 "reason": "coldiq returned no verdict"})), _fm():
            lead = await main.verify_revealed_lead(
                {"business_email": "a@b.com"}, "coldiq")

        self.assertEqual(lead["email_verification"]["checked_by"], "findymail")
        self.assertTrue(lead["email_verified"])
        # The reason ColdIQ could not answer is kept, not thrown away.
        self.assertIn("fell_back_from", lead["email_verification"])

    async def test_a_coldiq_verdict_is_not_second_guessed(self):
        FindymailClient.reset({})
        with patch.object(main.httpx, "AsyncClient", FindymailClient), \
             patch.object(main, "coldiq_verify_email", AsyncMock(return_value={
                 "status": "undeliverable", "sendable": False, "checked_by": "coldiq"})), _fm():
            lead = await main.verify_revealed_lead(
                {"business_email": "a@b.com"}, "coldiq")

        self.assertEqual(lead["email_verification"]["checked_by"], "coldiq")
        self.assertFalse(lead["email_verified"])
        self.assertEqual(FindymailClient.calls, [])

    async def test_no_email_costs_no_call_to_either(self):
        FindymailClient.reset({})
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            lead = await main.verify_revealed_lead({"contact_name": "Ada"}, "wiza")

        self.assertEqual(lead["email_verification"]["status"], "unverified")
        self.assertEqual(FindymailClient.calls, [])


class FindymailSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_poll_then_page(self):
        FindymailClient.reset({
            ("POST", "/api/intellimatch/search"): (200, {"hash": "abc123"}),
            ("GET", "/api/intellimatch/status"): (200, {"status": "success"}),
            ("GET", "/api/intellimatch/data"): (200, {
                "data": [{"name": "Acme", "contact_email": "a@acme.com",
                          "contact_name": "Ada Lovelace"}],
                "meta": {"total": 1}}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), \
             patch.object(main, "FINDYMAIL_POLL_INTERVAL", 0), _fm():
            data = await main.findymail_person_search({"semantic_query": "AI SaaS"}, 6)

        self.assertEqual(len(data["profiles"]), 1)
        self.assertEqual([c[1] for c in FindymailClient.calls],
                         ["/api/intellimatch/search", "/api/intellimatch/status",
                          "/api/intellimatch/data"])

    async def test_a_task_that_never_finishes_returns_what_it_has(self):
        # The credits are spent when the task runs, so partial results beat
        # nothing — and the hash stays collectable afterwards.
        FindymailClient.reset({
            ("POST", "/api/intellimatch/search"): (200, {"hash": "abc123"}),
            ("GET", "/api/intellimatch/status"): (200, {"status": "processing", "progress": 40}),
            ("GET", "/api/intellimatch/data"): (200, {"data": [{"name": "Acme"}], "meta": {"total": 9}}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), \
             patch.object(main, "FINDYMAIL_POLL_INTERVAL", 0), \
             patch.object(main, "FINDYMAIL_POLL_SECONDS", 0.01), _fm():
            data = await main.findymail_person_search({"semantic_query": "AI SaaS"}, 6)

        self.assertEqual(len(data["profiles"]), 1)

    async def test_a_failed_task_is_empty_rather_than_an_error(self):
        FindymailClient.reset({
            ("POST", "/api/intellimatch/search"): (200, {"hash": "abc123"}),
            ("GET", "/api/intellimatch/status"): (200, {"status": "failed"}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), \
             patch.object(main, "FINDYMAIL_POLL_INTERVAL", 0), _fm():
            data = await main.findymail_person_search({"semantic_query": "AI SaaS"}, 6)

        self.assertEqual(data["profiles"], [])
        # No point paging a task that failed.
        self.assertNotIn("/api/intellimatch/data", [c[1] for c in FindymailClient.calls])

    async def test_out_of_credits_is_a_402_the_caller_can_act_on(self):
        FindymailClient.reset({
            ("POST", "/api/intellimatch/search"): (402, {"error": "Not enough credits"}),
        })
        with patch.object(main.httpx, "AsyncClient", FindymailClient), _fm():
            with self.assertRaises(main.HTTPException) as caught:
                await main.findymail_person_search({"semantic_query": "AI SaaS"}, 6)

        self.assertEqual(caught.exception.status_code, 402)


class FindymailChainTests(unittest.TestCase):
    def test_it_sits_late_because_it_is_slow_and_billed_per_result(self):
        order = list(main.PROVIDER_ORDER)

        self.assertIn("findymail", order)
        self.assertGreater(order.index("findymail"), order.index("getleads"))
        self.assertGreater(order.index("findymail"), order.index("coldiq"))

    def test_no_key_removes_it_rather_than_breaking_the_chain(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "findymail_api_key", None), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertNotIn("findymail", main.provider_chain())

    def test_a_key_is_all_it_takes(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "findymail_api_key", "fm"), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertIn("findymail", main.provider_chain())


class FiberClient:
    """Fake httpx client for Fiber: answers per path, records headers and body."""
    routes = {}
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        path = url.replace(main.FIBER_BASE, "")
        self.__class__.calls.append((path, json, headers))
        status, payload = self.__class__.routes.get(path, (404, {}))
        return RoutedResponse(status, payload)

    @classmethod
    def reset(cls, routes=None):
        cls.routes = routes or {}
        cls.calls = []


def _fb(**env):
    return patch.multiple(main.settings, fiber_api_key="fb_key", **env)


class FiberSearchBodyTests(unittest.TestCase):
    def test_an_icp_without_a_size_uses_the_cheaper_people_search(self):
        path, body = main.build_fiber_search(
            {"job_title": "Founder", "industry": "computer software"}, 6)

        self.assertEqual(path, "/v1/people-search")
        self.assertEqual(body["pageSize"], 6)
        # includeCount costs an extra credit and the row count is what we paid
        # for anyway.
        self.assertIs(body["includeCount"], False)

    def test_a_size_moves_it_to_the_combined_endpoint(self):
        # people-search has no headcount field. Refusing would be honest and
        # would also make Fiber sit out nearly every search this product sends.
        path, body = main.build_fiber_search(
            {"job_title": "Founder", "company_size": "1-10"}, 6)

        self.assertEqual(path, "/v1/combined-search/paginated")
        self.assertEqual(
            body["companyConfig"]["searchParams"]["employeeCountV2"],
            {"lowerBoundExclusive": 0, "upperBoundInclusive": 10})

    def test_the_lower_bound_is_exclusive(self):
        # A 1-10 band starts above 0, not above 1 — off by one here silently
        # drops every one-person company.
        _, body = main.build_fiber_search({"job_title": "F", "company_size": "11-50"}, 6)
        bounds = body["companyConfig"]["searchParams"]["employeeCountV2"]

        self.assertEqual(bounds["lowerBoundExclusive"], 10)
        self.assertEqual(bounds["upperBoundInclusive"], 50)

    def test_a_country_is_sent_as_iso3(self):
        params = main.build_fiber_people_params({"job_title": "F", "location": "US"})
        self.assertEqual(params["country3LetterCode"], {"anyOf": ["USA"]})

    def test_a_state_carries_its_full_name(self):
        # "CA" alone is Canada to the classifier, not California.
        params = main.build_fiber_people_params(
            {"job_title": "F", "location": "California"})

        self.assertEqual(params["state"]["anyOf"][0]["stateName"], "California")
        self.assertEqual(params["state"]["anyOf"][0]["countryCode"], "USA")

    def test_a_city_gets_a_radius_because_the_field_requires_one(self):
        params = main.build_fiber_people_params(
            {"job_title": "F", "location": "San Francisco"})
        clause = params["location"]["unionAll"][0]

        self.assertEqual(clause["strategy"], "free-form-city")
        self.assertEqual(clause["radius"]["unit"], "miles")

    def test_a_continent_is_refused_rather_than_guessed(self):
        # Their presets are metro areas; none of them is Europe.
        with self.assertRaises(main.ProviderUnsupported):
            main.build_fiber_people_params({"job_title": "F", "location": "Europe"})

    def test_every_country_the_parser_emits_has_an_iso3(self):
        """The guard in the country branch must stay unreachable.

        classify_location returns an ISO-2 code drawn from _BM_COUNTRY_CODE. If
        a country is added there without an ISO-3 here, Fiber silently stops
        answering searches for it — a provider quietly sitting out a whole
        market is exactly the failure this chain keeps producing.
        """
        emitted = set(main._BM_COUNTRY_CODE.values())

        self.assertEqual(emitted - set(main._FIBER_COUNTRY3), set())

    def test_the_segment_term_becomes_free_text(self):
        params = main.build_fiber_people_params(
            {"job_title": "Founder", "semantic_keywords": "ai saas"})

        self.assertEqual(params["keywordsV2"]["clauses"][0]["terms"], ["ai saas"])


class FiberIndustryTests(unittest.TestCase):
    """An industry outside their enum is refused, never sent."""

    def test_a_shared_spelling_passes_through(self):
        self.assertEqual(main.fiber_industry("computer software"), "Computer Software")

    def test_their_newer_spelling_is_used_where_it_differs(self):
        self.assertEqual(main.fiber_industry("oil & energy"), "Oil and Gas")
        self.assertEqual(main.fiber_industry("pharmaceuticals"),
                         "Pharmaceutical Manufacturing")
        self.assertEqual(main.fiber_industry("hospital & health care"),
                         "Hospitals and Health Care")

    def test_an_unknown_industry_is_none(self):
        self.assertIsNone(main.fiber_industry("vertical ai agents"))

    def test_the_search_refuses_rather_than_zeroing_itself(self):
        # The Bytemine lesson: an unmapped industry does not error, it silently
        # matches nothing, and the chain reads that as "no such people".
        with self.assertRaises(main.ProviderUnsupported):
            main.build_fiber_people_params(
                {"job_title": "F", "industry": "vertical ai agents"})


class FiberSearchTests(unittest.IsolatedAsyncioTestCase):
    def _row(self):
        return {"name": "Ada Lovelace", "first_name": "Ada", "last_name": "Lovelace",
                "headline": "Founder at Acme", "primary_slug": "ada",
                "locality": "London", "industry_name": "Computer Software",
                "current_job": {"company_name": "Acme", "title": "Founder",
                                "seniority": "Executive"}}

    async def test_it_reads_the_people_search_envelope(self):
        FiberClient.reset({"/v1/people-search":
                           (200, {"output": {"data": [self._row()],
                                             "nextCursor": "c2"}})})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            data = await main.fiber_person_search({"job_title": "Founder"}, 6)

        self.assertEqual(data["total"], 1)
        self.assertEqual(data["next_cursor"], "c2")

    async def test_it_reads_the_combined_envelope_too(self):
        # The combined endpoint returns the same profile shape under a
        # different key, with its own cursor.
        FiberClient.reset({"/v1/combined-search/paginated":
                           (200, {"output": {"profiles": [self._row()],
                                             "nextProfilesCursor": "p2"}})})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            data = await main.fiber_person_search(
                {"job_title": "Founder", "company_size": "1-10"}, 6)

        self.assertEqual(data["total"], 1)
        self.assertEqual(data["next_cursor"], "p2")

    async def test_the_key_travels_in_the_header_not_the_body(self):
        # Every provider module here logs its request body, and a key in the
        # body is a key in the Railway logs.
        FiberClient.reset({"/v1/people-search": (200, {"output": {"data": []}})})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            await main.fiber_person_search({"job_title": "Founder"}, 6)

        _, body, headers = FiberClient.calls[0]
        self.assertEqual(headers["x-api-key"], "fb_key")
        self.assertNotIn("apiKey", body)

    async def test_an_error_is_an_empty_page_not_an_exception(self):
        FiberClient.reset({"/v1/people-search": (402, {"error": "no credits"})})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            data = await main.fiber_person_search({"job_title": "Founder"}, 6)

        self.assertEqual(data["profiles"], [])


class FiberTransformTests(unittest.TestCase):
    def test_the_documented_row_shape_maps_across(self):
        lead = main.transform_fiber_profile({
            "name": "Ada Lovelace", "first_name": "Ada", "last_name": "Lovelace",
            "headline": "Founder at Acme", "primary_slug": "ada-lovelace",
            "locality": "London", "industry_name": "Computer Software",
            "current_job": {"company_name": "Acme", "title": "Founder"},
        })

        self.assertEqual(lead["contact_name"], "Ada Lovelace")
        self.assertEqual(lead["job_title"], "Founder")
        self.assertEqual(lead["company_name"], "Acme")
        self.assertEqual(lead["linkedin_url"],
                         "https://www.linkedin.com/in/ada-lovelace")
        # Search returns no contact details — those are a separate, separately
        # billed reveal.
        self.assertIsNone(lead["business_email"])

    def test_the_already_seen_filter_can_read_this_row(self):
        # _profile_url has to find the URL or the repeat-lead filter matches
        # nothing — the defect that had ColdIQ showing the same people.
        lead = main.transform_fiber_profile({"name": "A", "primary_slug": "ada"})
        self.assertIsNotNone(main._profile_url(lead))


class FiberRevealTests(unittest.IsolatedAsyncioTestCase):
    def _payload(self, emails, phones=None):
        return {"output": {"profile": {"name": "Ada", "emails": emails,
                                       "phoneNumbers": phones or []}}}

    async def test_a_verified_address_wins_over_an_unverified_one(self):
        # Only `valid` has passed deliverability verification upstream.
        FiberClient.reset({"/v1/contact-details/single": (200, self._payload([
            {"email": "guess@acme.com", "type": "work", "status": "unknown"},
            {"email": "ada@acme.com", "type": "work", "status": "valid"},
        ]))})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            contact = await main.fiber_reveal("https://linkedin.com/in/ada")

        self.assertEqual(contact["email"], "ada@acme.com")
        self.assertEqual(contact["email_status"], "valid")

    async def test_personal_emails_are_never_paid_for(self):
        FiberClient.reset({"/v1/contact-details/single": (200, self._payload([]))})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            await main.fiber_reveal("https://linkedin.com/in/ada")

        sent = FiberClient.calls[0][1]["enrichmentType"]
        self.assertIs(sent["getWorkEmails"], True)
        self.assertIs(sent["getPersonalEmails"], False)

    async def test_phones_are_requested_so_the_leg_does_not_lose_them(self):
        # This leg returns early on a hit; without phones it would quietly drop
        # the number Wiza would have supplied for the same lead.
        FiberClient.reset({"/v1/contact-details/single": (200, self._payload(
            [{"email": "a@b.com", "type": "work", "status": "valid"}],
            [{"number": "+15551234567", "type": "mobile"}]))})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            contact = await main.fiber_reveal("https://linkedin.com/in/ada",
                                              want_phone=True)

        self.assertIs(FiberClient.calls[0][1]["enrichmentType"]["getPhoneNumbers"], True)
        self.assertEqual(contact["phone"], "+15551234567")

    async def test_no_url_is_no_call(self):
        FiberClient.reset({})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            self.assertEqual(await main.fiber_reveal(""), {})
        self.assertEqual(FiberClient.calls, [])


class FiberVerifyTests(unittest.IsolatedAsyncioTestCase):
    async def _verdict(self, payload, status=200):
        FiberClient.reset({"/v1/validate-email/single": (status, payload)})
        with patch.object(main.httpx, "AsyncClient", FiberClient), _fb():
            return await main.fiber_verify_email("ada@acme.com")

    async def test_ok_is_sendable(self):
        v = await self._verdict({"output": {"verdict": "ok", "is_catch_all": False,
                                            "is_role_based": False}})
        self.assertEqual(v["status"], "deliverable")
        self.assertIs(v["sendable"], True)
        self.assertIs(v["catch_all"], False)

    async def test_risky_is_a_real_answer_and_is_not_sendable(self):
        # A catch-all domain accepts everything; that is a verdict, not a
        # failure to reach one.
        v = await self._verdict({"output": {"verdict": "risky", "is_catch_all": True}})

        self.assertEqual(v["status"], "risky")
        self.assertIs(v["sendable"], False)
        self.assertIs(v["catch_all"], True)

    async def test_inconclusive_says_nobody_could_tell(self):
        v = await self._verdict({"output": {"verdict": "inconclusive"}})
        self.assertEqual(v["status"], "unknown")
        self.assertIsNone(v["sendable"])

    async def test_out_of_credits_does_not_invent_a_verdict(self):
        v = await self._verdict({"error": "no credits"}, status=402)
        self.assertIsNone(v["sendable"])
        self.assertIn("credits", v["reason"])


class FiberVerifierOrderTests(unittest.IsolatedAsyncioTestCase):
    """Fiber is asked before Findymail; ColdIQ is never second-guessed."""

    async def test_it_is_tried_before_findymail(self):
        asked: list = []

        async def coldiq(email):
            return {"status": "unknown", "sendable": None, "checked_by": "coldiq",
                    "reason": "coldiq out of credits"}

        async def fiber(email):
            asked.append("fiber")
            return {"status": "deliverable", "sendable": True, "checked_by": "fiber"}

        async def findymail(email):
            asked.append("findymail")
            return {"status": "deliverable", "sendable": True, "checked_by": "findymail"}

        with patch.object(main, "coldiq_verify_email", coldiq), \
             patch.object(main, "fiber_verify_email", fiber), \
             patch.object(main, "findymail_verify_email", findymail), \
             patch.object(main, "provider_configured", lambda n: True):
            lead = await main.verify_revealed_lead(
                {"business_email": "ada@acme.com"}, "wiza")

        self.assertEqual(asked, ["fiber"])
        self.assertEqual(lead["email_verification"]["checked_by"], "fiber")
        self.assertEqual(lead["email_verification"]["fell_back_from"],
                         "coldiq out of credits")

    async def test_a_coldiq_verdict_is_left_alone(self):
        asked: list = []

        async def coldiq(email):
            return {"status": "deliverable", "sendable": True, "checked_by": "coldiq"}

        async def fiber(email):
            asked.append("fiber")
            return {"status": "undeliverable", "sendable": False, "checked_by": "fiber"}

        with patch.object(main, "coldiq_verify_email", coldiq), \
             patch.object(main, "fiber_verify_email", fiber), \
             patch.object(main, "provider_configured", lambda n: True):
            lead = await main.verify_revealed_lead(
                {"business_email": "ada@acme.com"}, "wiza")

        self.assertEqual(asked, [])
        self.assertIs(lead["email_verified"], True)


class FiberChainTests(unittest.TestCase):
    def test_it_sits_after_getleads_and_before_the_reveal_tools(self):
        order = list(main.PROVIDER_ORDER)

        self.assertGreater(order.index("fiber"), order.index("getleads"))
        self.assertLess(order.index("fiber"), order.index("coldiq"))
        self.assertLess(order.index("fiber"), order.index("wiza"))

    def test_no_key_removes_it_rather_than_breaking_the_chain(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "fiber_api_key", None), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertNotIn("fiber", main.provider_chain())

    def test_a_key_is_all_it_takes(self):
        with patch.object(main.settings, "bytemine_api_key", "b"), \
             patch.object(main.settings, "fiber_api_key", "fb"), \
             patch.object(main.settings, "search_provider", "bytemine"):
            self.assertIn("fiber", main.provider_chain())


class ProviderFilterDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    """Turn "the search returns nothing" into "nothing once X is applied".

    Three fixes were aimed at the Bytemine and Crustdata zeros from a reading of
    the request bodies, and none of them moved it. A body that looks correct and
    a body that matches their index are not the same thing, and re-reading does
    not tell them apart.
    """

    def _probe(self, empties_on=None, seen=None):
        """A fake provider search that returns nothing once `empties_on` is set."""
        async def search(params, limit, **kwargs):
            if seen is not None:
                seen.append(sorted(params))
            if empties_on and params.get(empties_on):
                return {"profiles": [], "total": 0, "next_cursor": None}
            return {"profiles": [{"pid": "1"}], "total": 12, "next_cursor": None}
        return search

    async def diagnose(self, request, **searches):
        patches = [patch.object(main, "provider_configured", lambda n: True)]
        for name, attr in (("bytemine", "bytemine_person_search"),
                           ("crustdata", "crustdata_person_search"),
                           ("getleads", "getleads_person_search")):
            patches.append(patch.object(main, attr,
                                        searches.get(name, self._probe())))
        for p in patches:
            p.start()
        try:
            return await main.diagnose_provider_filters(request)
        finally:
            for p in patches:
                p.stop()

    async def test_it_names_the_filter_that_empties_the_search(self):
        report = await self.diagnose(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               company_size="1-10"),
            bytemine=self._probe(empties_on="industry"),
        )

        self.assertIn("industry", report["providers"]["bytemine"]["verdict"])
        steps = report["providers"]["bytemine"]["steps"]
        self.assertEqual([s["added"] for s in steps], ["job_title", "industry"])
        self.assertEqual(steps[0]["total"], 12)
        self.assertEqual(steps[1]["total"], 0)

    async def test_empty_on_the_first_filter_reads_as_the_whole_integration(self):
        report = await self.diagnose(
            main.SearchRequest(job_title="Founder", industry="computer software"),
            crustdata=self._probe(empties_on="job_title"),
        )

        self.assertIn("whole integration",
                      report["providers"]["crustdata"]["verdict"])

    async def test_filters_are_added_one_at_a_time(self):
        seen: list = []
        await self.diagnose(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               company_size="1-10"),
            getleads=self._probe(seen=seen),
        )

        # Each probe carries one more field than the last, and the walk runs to
        # the end when nothing empties it.
        added = [[f for f in call if f in main.DIAGNOSTIC_FILTER_ORDER] for call in seen]
        self.assertEqual(added, [["job_title"],
                                 ["industry", "job_title"],
                                 ["company_size", "industry", "job_title"]])

    async def test_a_provider_that_survives_every_filter_says_so(self):
        report = await self.diagnose(
            main.SearchRequest(job_title="Founder", industry="computer software"))

        self.assertEqual(report["providers"]["getleads"]["verdict"],
                         "no filter emptied it")

    async def test_a_refusal_is_reported_rather_than_raised(self):
        async def refuses(params, limit, **kwargs):
            raise main.ProviderUnsupported("location", params.get("location"))

        report = await self.diagnose(
            main.SearchRequest(job_title="Founder", location="US"),
            bytemine=refuses,
        )

        self.assertIn("unsupported_filter", report["providers"]["bytemine"]["verdict"])

    async def test_an_upstream_error_is_reported_rather_than_raised(self):
        async def fails(params, limit, **kwargs):
            raise main.HTTPException(status_code=402, detail="out of credits")

        report = await self.diagnose(
            main.SearchRequest(job_title="Founder"), crustdata=fails)

        step = report["providers"]["crustdata"]["steps"][0]
        self.assertEqual(step["outcome"], "error")
        self.assertEqual(step["status"], 402)

    async def test_it_is_reachable_from_a_url_bar(self):
        """A diagnostic nobody can run does not diagnose anything.

        The POST is the real shape, but running it needs a terminal or an HTTP
        client. The GET takes the same filters as query params so it can be
        pasted into a browser by whoever is watching the logs.
        """
        patches = [patch.object(main, "provider_configured", lambda n: True)]
        for attr in ("bytemine_person_search", "crustdata_person_search",
                     "getleads_person_search"):
            patches.append(patch.object(main, attr, self._probe(empties_on="industry")))
        for p in patches:
            p.start()
        try:
            report = await main.diagnose_provider_filters_from_url(
                job_title="Founder", industry="computer software")
        finally:
            for p in patches:
                p.stop()

        self.assertIn("industry", report["providers"]["bytemine"]["verdict"])
        self.assertEqual(report["params"]["job_title"], "Founder")

    async def test_a_search_with_nothing_to_isolate_is_refused(self):
        with self.assertRaises(main.HTTPException) as raised:
            await self.diagnose(main.SearchRequest(company_domain="acme.com"))

        self.assertEqual(raised.exception.status_code, 400)


class ProviderShapeProbeTests(unittest.IsolatedAsyncioTestCase):
    """The filter probe narrowed both zeros to something that is not a filter.

    Bytemine came back empty on a bare job title, and Crustdata returned
    4,353,682 for the same title and 0 the moment industry was added — so its
    "(.)" operator and field prefix are both fine, since the title filter uses
    both. This probe varies the request shape instead.
    """

    async def run_shapes(self, bytemine=None, crustdata=None, configured=("bytemine", "crustdata")):
        async def bm(path, body, timeout=60.0):
            return bytemine(body) if bytemine else {"contacts": [], "totalCount": 0}

        patches = [
            patch.object(main, "provider_configured", lambda n: n in configured),
            patch.object(main, "bytemine_call", bm),
        ]
        if crustdata is not None:
            patches.append(patch.object(main, "probe_crustdata_shape", crustdata))
        for p in patches:
            p.start()
        try:
            return await main.diagnose_provider_shapes()
        finally:
            for p in patches:
                p.stop()

    async def test_a_filterless_search_that_finds_people_blames_the_filter_keys(self):
        # The decisive probe: if the index answers with no filters at all, the
        # request is arriving and our filter names are what it cannot read.
        def bytemine(body):
            filtered = any(k in body for k in ("jobTitles", "job_titles", "titles", "jobTitle"))
            return {"contacts": [] if filtered else [{"pid": "1"}],
                    "totalCount": 0 if filtered else 900}

        report = await self.run_shapes(bytemine=bytemine, configured=("bytemine",))

        self.assertIn("filter keys", report["providers"]["bytemine"]["verdict"])

    async def test_a_filterless_search_that_finds_nothing_blames_the_request(self):
        report = await self.run_shapes(
            bytemine=lambda body: {"contacts": [], "totalCount": 0},
            configured=("bytemine",))

        self.assertIn("envelope", report["providers"]["bytemine"]["verdict"])

    async def test_the_empty_body_probe_really_is_empty(self):
        seen: list = []

        def bytemine(body):
            seen.append(body)
            return {"contacts": [], "totalCount": 0}

        await self.run_shapes(bytemine=bytemine, configured=("bytemine",))

        first = seen[0]
        self.assertEqual(set(first), {"pageSize", "page"})

    async def test_a_working_industry_spelling_is_named(self):
        async def crustdata(filters):
            field = filters["conditions"][1]["field"]
            works = field.endswith("company_professional_network_industry")
            return {"outcome": "ok", "total": 42 if works else 0,
                    "returned": 1 if works else 0}

        report = await self.run_shapes(crustdata=crustdata, configured=("crustdata",))
        verdict = report["providers"]["crustdata"]["verdict"]

        self.assertIn("company_professional_network_industry", verdict)

    async def test_no_working_spelling_says_so_rather_than_picking_one(self):
        async def crustdata(filters):
            return {"outcome": "ok", "total": 0, "returned": 0}

        report = await self.run_shapes(crustdata=crustdata, configured=("crustdata",))

        self.assertIn("no spelling", report["providers"]["crustdata"]["verdict"])

    async def test_an_unconfigured_provider_is_skipped(self):
        report = await self.run_shapes(configured=())
        self.assertEqual(report["providers"], {})


if __name__ == "__main__":
    unittest.main()
