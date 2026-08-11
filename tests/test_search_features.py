import json
import os
import unittest
from unittest.mock import patch

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
        self.assertEqual(
            by_field["experience.employment_details.current.company_industries"]["value"],
            "computer software",
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


class BytemineFilterTests(unittest.TestCase):
    def test_core_filters_map_onto_bytemine_fields(self):
        body = main.build_bytemine_filters({
            "job_title": "VP of Sales",
            "seniority": "vp",
            "industry": "Information Technology and Services",
            "company_size": "51-200",
            "company_domain": "stripe.com",
            "location": "CA",
        })

        self.assertEqual(body["jobTitles"], ["VP of Sales"])
        self.assertEqual(body["seniorityLevels"], ["VP"])
        self.assertEqual(body["industries"], ["Information Technology and Services"])
        self.assertEqual(body["employeeSizes"], ["51-200"])
        self.assertEqual(body["urls"], ["stripe.com"])
        self.assertEqual(body["states"], ["CA"])

    def test_founder_maps_to_the_owner_seniority(self):
        body = main.build_bytemine_filters({"job_title": "Founder", "seniority": "founder"})
        self.assertEqual(body["seniorityLevels"], ["Owner"])

    def test_c_suite_maps_to_the_canonical_c_team_spelling(self):
        for value in ("cxo", "c-level", "c-suite", "executive"):
            body = main.build_bytemine_filters({"job_title": "CEO", "seniority": value})
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
                    {"job_title": "VP of Sales", "location": "CA"}, 25)
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

    async def run_search(self, request, responses):
        """Run /search with the cache disabled and each provider call stubbed.

        `responses` maps a provider name to either the value its fetcher should
        return or an exception it should raise. A provider left out of the map
        is not stubbed at all, so its real implementation runs — which is how the
        genuine "this provider cannot express that filter" refusal gets covered.
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
            patch.object(main.settings, "crustdata_api_key", "c"),
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

    async def test_bytemine_serves_the_search_when_it_can(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": {"profiles": [{"pid": "1", "first_name": "Ada"}],
                          "total": 1, "next_cursor": None},
             "crustdata": AssertionError("must not be reached")},
        )

        self.assertEqual(calls, ["bytemine"])
        self.assertEqual(response.provider, "bytemine")
        self.assertEqual(response.count, 1)
        self.assertEqual(response.leads[0]["contact_name"], "Ada")

    async def test_a_filter_bytemine_cannot_express_moves_to_crustdata(self):
        # Keywords have no field on /contacts/search. Falling through keeps the
        # ICP whole; dropping the keyword would not.
        response, calls = await self.run_search(
            main.SearchRequest(job_title="Founder", industry="computer software",
                               keywords="pre-revenue"),
            # Bytemine is deliberately not stubbed: the real
            # build_bytemine_filters must be the thing that refuses.
            {"crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(calls, ["crustdata"])
        self.assertEqual(response.provider, "crustdata")
        self.assertEqual(
            [a["outcome"] for a in response.provider_attempts], ["unsupported_filter"])

    async def test_an_exhausted_bytemine_balance_falls_through(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": HTTPException(status_code=402, detail="no credits"),
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(calls, ["bytemine", "crustdata"])
        self.assertEqual(response.provider, "crustdata")
        self.assertEqual(response.provider_attempts[0]["detail"], 402)

    async def test_no_results_from_the_leader_still_tries_the_next(self):
        response, calls = await self.run_search(
            main.SearchRequest(job_title="VP of Sales", location="CA"),
            {"bytemine": {"profiles": [], "total": 0, "next_cursor": None},
             "crustdata": {"profiles": [{"crustdata_person_id": 9}],
                           "total": 1, "next_cursor": None}},
        )

        self.assertEqual(calls, ["bytemine", "crustdata"])
        self.assertEqual(response.provider, "crustdata")
        self.assertEqual(response.provider_attempts[0]["outcome"], "no_results")

    async def test_a_bad_request_is_not_retried_against_every_provider(self):
        # A 400 is about the query, not the provider — it will fail identically
        # everywhere, so it surfaces instead of burning a call on each one.
        with self.assertRaises(HTTPException) as ctx:
            await self.run_search(
                main.SearchRequest(job_title="VP of Sales", location="CA"),
                {"bytemine": HTTPException(status_code=400, detail="bad filter"),
                 "crustdata": AssertionError("must not be reached")},
            )
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_the_last_providers_failure_is_reported(self):
        # Nothing is left to fall back to, so the error must reach the caller
        # rather than being swallowed into an empty result set.
        with self.assertRaises(HTTPException) as ctx:
            await self.run_search(
                main.SearchRequest(job_title="VP of Sales", location="CA"),
                {"bytemine": HTTPException(status_code=402, detail="no credits"),
                 "crustdata": HTTPException(status_code=500, detail="upstream down"),
                 "wiza": HTTPException(status_code=503, detail="wiza down")},
            )
        self.assertEqual(ctx.exception.status_code, 503)


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
                        {"job_title": "CEO", "location": "CA"}, 10, offset=offset)
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


if __name__ == "__main__":
    unittest.main()
