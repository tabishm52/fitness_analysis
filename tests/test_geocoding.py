"""Tests for reverse geocoding and its persistent SQLite cache."""

from fitness_analysis import cache_db, geocoding


class FakeProvider:
    """Duck-typed geocoding provider: no network, records every call."""

    name = "fake"

    def __init__(self, addresses=None, coords=None):
        self.addresses = addresses or {}
        self.coords = coords or {}
        self.reverse_calls = []
        self.geocode_calls = []

    def geocode(self, address):
        self.geocode_calls.append(address)
        return self.coords.get(address)

    def reverse(self, lat, lon):
        pos = geocoding.round_pos((lat, lon))
        self.reverse_calls.append(pos)
        return self.addresses.get(pos)


def _from_env_raises(monkeypatch):
    """Fails the test if GeocodingProvider.from_env is ever reached (i.e. no network)."""

    def _raise(*_args, **_kwargs):
        raise AssertionError("GeocodingProvider.from_env should not be called")

    monkeypatch.setattr(geocoding.GeocodingProvider, "from_env", _raise)


# --------------------------------------------------------------------------------------
# round_pos
# --------------------------------------------------------------------------------------


def test_round_pos_rounds_to_four_decimals():
    assert geocoding.round_pos((37.774912345, -122.419412345)) == (37.7749, -122.4194)


# --------------------------------------------------------------------------------------
# GeocodingProvider.from_env
# --------------------------------------------------------------------------------------


def test_from_env_uses_google_when_api_key_set(monkeypatch):
    monkeypatch.setenv("TEST_GOOGLE_API_KEY", "fake-key")
    provider = geocoding.GeocodingProvider.from_env("TEST_GOOGLE_API_KEY")
    assert provider.name == "google"


def test_from_env_uses_nominatim_when_no_api_key(monkeypatch):
    monkeypatch.delenv("TEST_GOOGLE_API_KEY", raising=False)
    provider = geocoding.GeocodingProvider.from_env("TEST_GOOGLE_API_KEY")
    assert provider.name == "nominatim"


# --------------------------------------------------------------------------------------
# store_geocode_cache / lookup_geocode_cache
# --------------------------------------------------------------------------------------


def test_lookup_geocode_cache_exact_match(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "123 Main St", "nominatim")
        found, addr = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)

    assert found is True
    assert addr == "123 Main St"


def test_lookup_geocode_cache_proximity_match_within_radius(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "123 Main St", "nominatim")
        # ~11 m away (one 4-decimal step in latitude), well inside the 200 m radius.
        found, addr = geocoding.lookup_geocode_cache(db, (37.7750, -122.4194), 200.0)

    assert found is True
    assert addr == "123 Main St"


def test_lookup_geocode_cache_outside_radius_is_a_miss(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "123 Main St", "nominatim")
        # ~1.1 km away, outside the 200 m radius.
        found, addr = geocoding.lookup_geocode_cache(db, (37.7849, -122.4194), 200.0)

    assert found is False
    assert addr is None


def test_lookup_geocode_cache_true_miss_returns_not_found(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        found, addr = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)

    assert found is False
    assert addr is None


def test_lookup_geocode_cache_distinguishes_negative_result_from_miss(tmp_path):
    """A cached "provider found nothing" entry must read back as found=True."""
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), None, "nominatim")
        cached_negative = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)
        true_miss = geocoding.lookup_geocode_cache(db, (10.0, 10.0), 200.0)

    assert cached_negative == (True, None)
    assert true_miss == (False, None)


def test_store_geocode_cache_replaces_existing_entry(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "Old Address", "nominatim")
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "New Address", "google")
        found, addr = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)

    assert found is True
    assert addr == "New Address"


# --------------------------------------------------------------------------------------
# invalidate_geocode_cache
# --------------------------------------------------------------------------------------


def test_invalidate_geocode_cache_missing_db_is_noop(tmp_path):
    geocoding.invalidate_geocode_cache(tmp_path)  # should not raise


def test_invalidate_geocode_cache_none_cache_dir_is_noop(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "123 Main St", "nominatim")

    geocoding.invalidate_geocode_cache(None)  # should not raise

    with cache_db.open_db(tmp_path) as db:
        found, _ = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)
    assert found is True


def test_invalidate_geocode_cache_none_clears_everything_and_fingerprint(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "123 Main St", "nominatim")
        with db.conn:
            db["cluster_fingerprints"].insert({"table_name": "activities", "fingerprint": "abc"})

    geocoding.invalidate_geocode_cache(tmp_path)

    with cache_db.open_db(tmp_path) as db:
        found, _ = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)
        assert found is False
        assert list(db["cluster_fingerprints"].rows) == []


def test_invalidate_geocode_cache_filters_by_provider(tmp_path):
    with cache_db.open_db(tmp_path) as db:
        geocoding.store_geocode_cache(db, (37.7749, -122.4194), "Nominatim Addr", "nominatim")
        geocoding.store_geocode_cache(db, (10.0, 10.0), "Google Addr", "google")

    geocoding.invalidate_geocode_cache(tmp_path, provider="nominatim")

    with cache_db.open_db(tmp_path) as db:
        nominatim_found, _ = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)
        google_found, google_addr = geocoding.lookup_geocode_cache(db, (10.0, 10.0), 200.0)
    assert nominatim_found is False
    assert google_found is True
    assert google_addr == "Google Addr"


# --------------------------------------------------------------------------------------
# seed_geocode_cache
# --------------------------------------------------------------------------------------


def test_seed_geocode_cache_missing_cache_dir_is_noop(monkeypatch):
    _from_env_raises(monkeypatch)
    geocoding.seed_geocode_cache(["123 Main St"], None)  # should not raise


def test_seed_geocode_cache_uses_injected_provider_no_network(tmp_path, monkeypatch):
    _from_env_raises(monkeypatch)
    provider = FakeProvider(coords={"123 Main St": (37.7749, -122.4194)})
    config = geocoding.GeocodingConfig(provider=provider)

    geocoding.seed_geocode_cache(["123 Main St"], tmp_path, config)

    assert provider.geocode_calls == ["123 Main St"]
    with cache_db.open_db(tmp_path) as db:
        found, addr = geocoding.lookup_geocode_cache(db, (37.7749, -122.4194), 200.0)
    assert found is True
    assert addr == "123 Main St"


def test_seed_geocode_cache_skips_unresolved_addresses(tmp_path, monkeypatch):
    _from_env_raises(monkeypatch)
    provider = FakeProvider(coords={})
    config = geocoding.GeocodingConfig(provider=provider)

    geocoding.seed_geocode_cache(["Nonexistent Address"], tmp_path, config)

    with cache_db.open_db(tmp_path) as db:
        assert list(db["geocode_cache"].rows) == []


# --------------------------------------------------------------------------------------
# geocode_positions
# --------------------------------------------------------------------------------------


def test_geocode_positions_no_cache_dir_uses_injected_provider(monkeypatch):
    _from_env_raises(monkeypatch)
    provider = FakeProvider(addresses={(37.7749, -122.4194): "123 Main St"})
    config = geocoding.GeocodingConfig(provider=provider)

    result = geocoding.geocode_positions([(37.7749, -122.4194)], None, config)

    assert result == {(37.7749, -122.4194): "123 Main St"}
    assert provider.reverse_calls == [(37.7749, -122.4194)]


def test_geocode_positions_no_cache_dir_dedups_positions(monkeypatch):
    _from_env_raises(monkeypatch)
    provider = FakeProvider(addresses={(37.7749, -122.4194): "123 Main St"})
    config = geocoding.GeocodingConfig(provider=provider)

    geocoding.geocode_positions([(37.7749, -122.4194), (37.7749, -122.4194)], None, config)

    assert provider.reverse_calls == [(37.7749, -122.4194)]


def test_geocode_positions_caches_across_calls(tmp_path, monkeypatch):
    _from_env_raises(monkeypatch)
    provider = FakeProvider(addresses={(37.7749, -122.4194): "123 Main St"})
    config = geocoding.GeocodingConfig(provider=provider)
    pos = (37.7749, -122.4194)

    first = geocoding.geocode_positions([pos], tmp_path, config)
    second = geocoding.geocode_positions([pos], tmp_path, config)

    assert first == {pos: "123 Main St"}
    assert second == {pos: "123 Main St"}
    assert provider.reverse_calls == [pos]


def test_geocode_positions_negative_result_not_requeried(tmp_path, monkeypatch):
    """A position with no resolvable address must be cached too, not re-queried."""
    _from_env_raises(monkeypatch)
    provider = FakeProvider(addresses={})
    config = geocoding.GeocodingConfig(provider=provider)
    pos = (37.7749, -122.4194)

    first = geocoding.geocode_positions([pos], tmp_path, config)
    second = geocoding.geocode_positions([pos], tmp_path, config)

    assert first == {pos: None}
    assert second == {pos: None}
    assert provider.reverse_calls == [pos]
