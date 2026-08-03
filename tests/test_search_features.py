import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/test")
os.environ.setdefault("WIZA_API_KEY", "test-wiza")
os.environ.setdefault("CRUSTDATA_API_KEY", "test-crustdata")

import main


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


if __name__ == "__main__":
    unittest.main()
