"""Unit tests for KaggleDatasetSource. No live Kaggle calls."""

from __future__ import annotations

import httpx
import pytest

from app.schemas.dataset import SourceSearchRequest, SourceType, VerificationStatus
from app.services.datasets.base import DatasetSourceError
from app.services.datasets.kaggle_source import KaggleDatasetSource
from tests.conftest import (
    make_kaggle_payload,
    make_kaggle_record,
    make_mock_transport,
    make_settings,
)


def _source(responses, **setting_overrides) -> KaggleDatasetSource:
    """A Kaggle source wired to a mock transport replaying `responses`."""
    client = httpx.AsyncClient(transport=make_mock_transport(responses))
    return KaggleDatasetSource(
        settings=make_settings(**setting_overrides), client=client
    )


def _request(query: str = "student academic performance", limit: int = 10):
    return SourceSearchRequest(query=query, limit=limit)


@pytest.mark.asyncio
class TestKaggleSuccessfulSearch:
    async def test_returns_normalized_candidates(self):
        source = _source([httpx.Response(200, json=make_kaggle_payload(3))])
        candidates = await source.search(_request())

        assert len(candidates) == 3
        assert all(c.source == "kaggle" for c in candidates)
        assert all(c.source_type is SourceType.VERIFIED_EXTERNAL for c in candidates)
        assert all(c.verification_status is VerificationStatus.VERIFIED for c in candidates)

    async def test_maps_published_metadata(self):
        payload = make_kaggle_record(
            ref="jaya/student-habits",
            title="Student Habits vs Academic Performance",
            subtitle="Survey of study habits",
            downloads=70382,
            votes=886,
            usability=1.0,
            license_name="Apache 2.0",
            total_bytes=19512,
            last_updated="2025-04-12T10:49:08.663Z",
        )
        source = _source([httpx.Response(200, json=[payload])])
        candidate = (await source.search(_request()))[0]

        assert candidate.source_id == "jaya/student-habits"
        assert candidate.id == "kaggle:jaya/student-habits"
        assert candidate.name == "Student Habits vs Academic Performance"
        assert candidate.url == "https://www.kaggle.com/datasets/jaya/student-habits"
        assert candidate.download_count == 70382
        assert candidate.vote_count == 886
        assert candidate.usability == 1.0
        assert candidate.size_bytes == 19512
        assert candidate.license == "Apache 2.0"
        assert candidate.updated_at is not None
        assert candidate.updated_at.year == 2025
        assert candidate.description == "Survey of study habits"
        assert "education" in candidate.tags
        assert candidate.domain == "education"

    async def test_never_fabricates_unavailable_metadata(self):
        """Kaggle's list endpoint exposes no creation date and an empty file
        list. Those must stay None/empty rather than being invented."""
        source = _source([httpx.Response(200, json=[make_kaggle_record()])])
        candidate = (await source.search(_request()))[0]

        assert candidate.created_at is None
        assert candidate.file_count is None
        assert candidate.file_types == []

    async def test_respects_limit(self):
        source = _source([httpx.Response(200, json=make_kaggle_payload(5))])
        candidates = await source.search(_request(limit=2))
        assert len(candidates) == 2

    async def test_limit_is_clamped_to_source_maximum(self):
        source = _source([httpx.Response(200, json=make_kaggle_payload(5))])
        candidates = await source.search(_request(limit=9999))
        assert len(candidates) <= KaggleDatasetSource.max_limit

    async def test_sends_query_and_page_params(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=make_kaggle_payload(1))

        source = KaggleDatasetSource(
            settings=make_settings(),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await source.search(_request())

        assert seen[0].url.params["search"] == "student academic performance"
        assert seen[0].url.params["page"] == "1"
        assert seen[0].url.path.endswith("/datasets/list")

    async def test_does_not_download_dataset_files(self):
        """Discovery must never hit a download or files endpoint."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json=make_kaggle_payload(1))

        source = KaggleDatasetSource(
            settings=make_settings(),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await source.search(_request())

        assert seen == ["/api/v1/datasets/list"]

    async def test_pagination_stops_at_max_pages(self):
        """One page per request, capped, so a large limit cannot loop forever."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                200, json=[make_kaggle_record(ref=f"owner/dataset-{calls['n']}")]
            )

        source = KaggleDatasetSource(
            settings=make_settings(kaggle_max_pages=2),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await source.search(_request(limit=50))

        assert calls["n"] == 2

    async def test_pagination_stops_on_empty_page(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=[make_kaggle_record()])
            return httpx.Response(200, json=[])

        source = KaggleDatasetSource(
            settings=make_settings(kaggle_max_pages=3),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        candidates = await source.search(_request(limit=50))

        assert calls["n"] == 2
        assert len(candidates) == 1


@pytest.mark.asyncio
class TestKaggleEmptyResult:
    async def test_empty_list_returns_no_candidates(self):
        source = _source([httpx.Response(200, json=[])])
        assert await source.search(_request()) == []

    async def test_empty_body_is_not_an_error(self):
        source = _source([httpx.Response(200, json={})])
        assert await source.search(_request()) == []


@pytest.mark.asyncio
class TestKaggleAuthenticationFailure:
    @pytest.mark.parametrize("status", [401, 403])
    async def test_raises_authentication_error(self, status):
        source = _source([httpx.Response(status, json={"message": "unauthorized"})])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_AUTHENTICATION_ERROR"

    async def test_error_message_never_contains_credentials(self):
        source = _source([httpx.Response(401, json={"message": "bad token"})])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert "token" not in exc.value.message.lower().replace("credentials", "")


@pytest.mark.asyncio
class TestKaggleRateLimit:
    async def test_429_raises_rate_limited(self):
        source = _source([httpx.Response(429, text="slow down")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_RATE_LIMITED"


@pytest.mark.asyncio
class TestKaggleTimeout:
    async def test_timeout_raises_timeout_error(self):
        source = _source([httpx.TimeoutException("timed out")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_TIMEOUT"

    async def test_connection_error_raises_unavailable(self):
        source = _source([httpx.ConnectError("no route")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_UNAVAILABLE"

    async def test_server_error_raises_unavailable(self):
        source = _source([httpx.Response(503, text="maintenance")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_UNAVAILABLE"

    async def test_unexpected_status_is_sanitized(self):
        source = _source([httpx.Response(418, text="I am a teapot")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_UNAVAILABLE"
        assert "teapot" not in exc.value.message


@pytest.mark.asyncio
class TestKaggleMalformedResult:
    async def test_non_json_body_raises_invalid_response(self):
        source = _source([httpx.Response(200, text="<html>error</html>")])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(_request())
        assert exc.value.code == "KAGGLE_INVALID_RESPONSE"

    async def test_wrong_shape_returns_no_candidates(self):
        source = _source([httpx.Response(200, json={"unexpected": "object"})])
        assert await source.search(_request()) == []

    async def test_malformed_rows_are_skipped_not_fatal(self):
        payload = [
            make_kaggle_record(ref="owner/good"),
            {"title": None, "ref": None, "subtitle": None},
            "not-a-dict",
            make_kaggle_record(ref="owner/good2"),
        ]
        source = _source([httpx.Response(200, json=payload)])
        candidates = await source.search(_request())

        assert len(candidates) == 2
        assert [c.source_id for c in candidates] == ["owner/good", "owner/good2"]

    async def test_rows_with_null_fields_do_not_crash(self):
        payload = [
            {
                "ref": "owner/minimal",
                "titleNullable": "Minimal Dataset",
                "subtitleNullable": None,
                "descriptionNullable": None,
                "urlNullable": None,
                "ownerRefNullable": None,
                "licenseNameNullable": None,
                "totalBytesNullable": None,
                "lastUpdatedNullable": None,
                "downloadCountNullable": None,
                "voteCountNullable": None,
                "usabilityRatingNullable": None,
                "tagsNullable": None,
            }
        ]
        source = _source([httpx.Response(200, json=payload)])
        candidates = await source.search(_request())

        assert len(candidates) == 1
        assert candidates[0].description is None
        assert candidates[0].download_count is None
        assert candidates[0].usability is None
        assert candidates[0].tags == []
        assert candidates[0].updated_at is None


@pytest.mark.asyncio
class TestKaggleValidation:
    async def test_empty_query_raises(self):
        source = _source([httpx.Response(200, json=[])])
        with pytest.raises(DatasetSourceError) as exc:
            await source.search(SourceSearchRequest(query="   ", limit=5))
        assert exc.value.code == "DATASET_QUERY_EMPTY"

    async def test_get_metadata_requires_owner_slug_ref(self):
        source = _source([httpx.Response(200, json=[])])
        with pytest.raises(DatasetSourceError) as exc:
            await source.get_metadata("just-a-slug")
        assert exc.value.code == "DATASET_SOURCE_INVALID_ID"

    async def test_get_metadata_returns_matching_candidate(self):
        source = _source([httpx.Response(200, json=[make_kaggle_record(ref="owner/student-performance")])])
        candidate = await source.get_metadata("owner/student-performance")
        assert candidate.source_id == "owner/student-performance"


class TestKaggleConfiguration:

    def test_is_configured_when_enabled(self):
        source = KaggleDatasetSource(settings=make_settings(kaggle_enabled=True))
        assert source.is_configured is True

    def test_not_configured_when_disabled(self):
        source = KaggleDatasetSource(settings=make_settings(kaggle_enabled=False))
        assert source.is_configured is False

    def test_repr_does_not_leak_token(self):
        source = KaggleDatasetSource(settings=make_settings(kaggle_api_token="KGAT_secret"))
        assert "KGAT_secret" not in repr(source)
        assert "token_present=True" in repr(source)

    @pytest.mark.asyncio
    async def test_token_is_sent_as_bearer_header_not_query_param(self):
        seen: list[httpx.Request] = []


        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=make_kaggle_payload(1))

        source = KaggleDatasetSource(
            settings=make_settings(kaggle_api_token="KGAT_secret_value"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await source.search(_request())

        assert seen[0].headers["authorization"] == "Bearer KGAT_secret_value"
        assert "KGAT_secret_value" not in str(seen[0].url)

    @pytest.mark.asyncio
    async def test_omits_authorization_header_when_no_token(self):

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=make_kaggle_payload(1))

        source = KaggleDatasetSource(
            settings=make_settings(kaggle_api_token=""),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        await source.search(_request())

        assert "authorization" not in seen[0].headers

