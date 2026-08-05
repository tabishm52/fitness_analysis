"""Tests for GPS route clustering."""

from fitness_analysis import geocoding, routes


def test_compute_cluster_fingerprint_identical_with_and_without_provider():
    """Companion to the provider seam: a live provider must not affect the fingerprint,
    and asdict-ing it (which deep-copies non-dataclass fields) must not raise."""
    keys = [("a.gpx", -1)]
    fp_no_provider = routes.compute_cluster_fingerprint(keys, routes.RouteClusterConfig())

    # A real (non-fake) provider: its RateLimiter wraps a thread lock, which
    # copy.deepcopy cannot handle, so this exercises the actual failure mode.
    provider = geocoding.GeocodingProvider.from_env("FITNESS_ANALYSIS_TEST_UNSET_VAR")
    config_with_provider = routes.RouteClusterConfig(
        geocoding=geocoding.GeocodingConfig(provider=provider)
    )
    fp_with_provider = routes.compute_cluster_fingerprint(keys, config_with_provider)

    assert fp_no_provider == fp_with_provider
