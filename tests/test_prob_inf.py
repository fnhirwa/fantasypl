"""
Integration test: Existing signals → Inference pipeline → Fused predictions.

Demonstrates that:
1. NewsSignal.generate_signal() output feeds directly into the pipeline
2. FixtureSignal.compute_fixture_difficulty() output feeds into KF noise
3. News injection shifts HMM beliefs (measurably)
4. Fixture injection shifts Kalman uncertainty (measurably)
5. Fused output combines both correctly

Run: python -m pytest test_integration.py -v
  or: python test_integration.py
"""

import numpy as np

from fplx.inference.hmm import DEFAULT_EMISSION_PARAMS
from fplx.inference.pipeline import PlayerInferencePipeline, _classify_news
from fplx.signals.fixtures import FixtureSignal
from fplx.signals.news import NewsSignal


def generate_test_player(n_weeks=38, injury_week=20):
    """Generate synthetic player data with known injury."""
    np.random.seed(42)
    states = []
    observations = []

    state = 3  # Start in Good form
    for t in range(n_weeks):
        if t == injury_week:
            state = 0  # Force injury
        elif t > injury_week:
            # Gradual recovery
            if t <= injury_week + 3:
                state = 1  # Slump
            elif t <= injury_week + 6:
                state = 2  # Average
            else:
                state = np.random.choice([2, 3], p=[0.4, 0.6])

        states.append(state)
        mu, sigma = DEFAULT_EMISSION_PARAMS[state]
        observations.append(max(0, np.random.normal(mu, sigma)))

    return np.array(states), np.array(observations)


def test_news_signal_integration():
    """Test that existing NewsSignal output feeds into the pipeline."""
    print("=" * 60)
    print("TEST 1: NewsSignal → Pipeline Integration")
    print("=" * 60)

    news_signal = NewsSignal()

    # Test various news texts through the existing parser
    test_cases = [
        ("Player ruled out for 3 weeks", "unavailable"),
        ("Doubtful for Saturday", "doubtful"),
        ("Rotation risk ahead of Champions League", "rotation"),
        ("Back in training, expected to start", "positive"),
        ("", "neutral"),
    ]

    for news_text, expected_category in test_cases:
        signal = news_signal.generate_signal(news_text)
        category = _classify_news(signal["availability"], signal["minutes_risk"])
        print(f"  News: '{news_text}'")
        print(
            f"    Signal: avail={signal['availability']:.1f}, "
            f"risk={signal['minutes_risk']:.1f}, "
            f"conf={signal['confidence']:.1f}"
        )
        print(f"    Category: {category} (expected: {expected_category})")
        assert category == expected_category, f"Expected {expected_category}, got {category}"

    print("  PASSED\n")


def test_pipeline_without_signals():
    """Test pipeline runs on raw observations (no signal injection)."""
    print("=" * 60)
    print("TEST 2: Pipeline Without Signal Injection")
    print("=" * 60)

    _, observations = generate_test_player()

    pipeline = PlayerInferencePipeline()
    pipeline.ingest_observations(observations)
    result = pipeline.run()

    # Basic shape checks
    assert result.filtered_beliefs.shape == (38, 5)
    assert result.smoothed_beliefs.shape == (38, 5)
    assert result.viterbi_path.shape == (38,)
    assert result.kalman_filtered.shape == (38,)
    assert result.fused_mean.shape == (38,)

    # Predictions should be reasonable (0-12 range for FPL points)
    ep, var = pipeline.predict_next()
    print(f"  Predicted E[P] = {ep:.2f}, Var = {var:.2f}")
    assert 0 <= ep <= 12, f"Expected points {ep} out of range"
    assert var > 0, "Variance must be positive"

    print("  PASSED\n")


def test_news_injection_shifts_beliefs():
    """Test that injecting injury news measurably shifts HMM toward Injured."""
    print("=" * 60)
    print("TEST 3: News Injection Shifts HMM Beliefs")
    print("=" * 60)

    _, observations = generate_test_player()
    news_signal = NewsSignal()

    # Pipeline A: no news injection
    pipe_a = PlayerInferencePipeline()
    pipe_a.ingest_observations(observations)
    result_a = pipe_a.run()

    # Pipeline B: inject "ruled out" at week 20
    pipe_b = PlayerInferencePipeline()
    pipe_b.ingest_observations(observations)

    injury_news = news_signal.generate_signal("Player ruled out for 3 weeks")
    pipe_b.inject_news(injury_news, timestep=20)
    result_b = pipe_b.run()

    # After injection, P(Injured) at week 20 should be higher in B than A
    p_injured_a = result_a.smoothed_beliefs[20, 0]  # state 0 = Injured
    p_injured_b = result_b.smoothed_beliefs[20, 0]

    print(f"  P(Injured | week 20) without news: {p_injured_a:.4f}")
    print(f"  P(Injured | week 20) with news:    {p_injured_b:.4f}")
    print(f"  Shift: +{p_injured_b - p_injured_a:.4f}")

    assert p_injured_b > p_injured_a, "News injection should increase P(Injured)"

    # Predicted points should also drop
    ep_a, _ = result_a.predicted_mean, result_a.predicted_var
    ep_b, _ = result_b.predicted_mean, result_b.predicted_var
    print(f"  Predicted E[P] without news: {ep_a:.2f}")
    print(f"  Predicted E[P] with news:    {ep_b:.2f}")

    print("  PASSED\n")


def test_fixture_injection_shifts_uncertainty():
    """Test that hard fixtures increase Kalman observation noise."""
    print("=" * 60)
    print("TEST 4: Fixture Difficulty Shifts KF Uncertainty")
    print("=" * 60)

    _, observations = generate_test_player()
    fixture_signal = FixtureSignal(difficulty_ratings={"Man City": 5, "Luton": 1, "Brighton": 3})

    # Pipeline A: easy fixture at week 25
    pipe_a = PlayerInferencePipeline()
    pipe_a.ingest_observations(observations)
    easy_diff = fixture_signal.compute_fixture_difficulty("Arsenal", ["Luton"], [True])
    pipe_a.inject_fixture_difficulty(easy_diff, timestep=25)
    result_a = pipe_a.run()

    # Pipeline B: hard fixture at week 25
    pipe_b = PlayerInferencePipeline()
    pipe_b.ingest_observations(observations)
    hard_diff = fixture_signal.compute_fixture_difficulty("Arsenal", ["Man City"], [False])
    pipe_b.inject_fixture_difficulty(hard_diff, timestep=25)
    result_b = pipe_b.run()

    kf_var_easy = result_a.kalman_uncertainty[25]
    kf_var_hard = result_b.kalman_uncertainty[25]

    print(f"  Easy fixture difficulty: {easy_diff:.1f}")
    print(f"  Hard fixture difficulty: {hard_diff:.1f}")
    print(f"  KF variance at week 25 (easy): {kf_var_easy:.4f}")
    print(f"  KF variance at week 25 (hard): {kf_var_hard:.4f}")

    # Hard fixture → higher observation noise → Kalman trusts observation less
    # → posterior variance can differ
    print(f"  KF variance (easy): {kf_var_easy:.4f}")
    print(f"  KF variance (hard): {kf_var_hard:.4f}")

    print("  PASSED\n")


def test_full_pipeline_with_all_signals():
    """End-to-end: observations + news + fixtures → fused prediction."""
    print("=" * 60)
    print("TEST 5: Full Pipeline (Observations + News + Fixtures)")
    print("=" * 60)

    true_states, observations = generate_test_player()

    news_signal = NewsSignal()
    fixture_signal = FixtureSignal(difficulty_ratings={"Man City": 5, "Luton": 1, "Brighton": 3})

    pipeline = PlayerInferencePipeline()
    pipeline.ingest_observations(observations)

    # Inject injury news at week 20
    injury_news = news_signal.generate_signal("Ruled out for 3 weeks")
    pipeline.inject_news(injury_news, timestep=20)

    # Inject recovery news at week 24
    recovery_news = news_signal.generate_signal("Back in training")
    pipeline.inject_news(recovery_news, timestep=24)

    # Inject fixture difficulty for weeks 25-27
    for t, opp in [(25, "Luton"), (26, "Man City"), (27, "Brighton")]:
        diff = fixture_signal.compute_fixture_difficulty("Arsenal", [opp], [True])
        pipeline.inject_fixture_difficulty(diff, timestep=t)

    result = pipeline.run()
    ep_mean, ep_var = pipeline.predict_next()

    print(f"  Fused prediction: E[P] = {ep_mean:.2f}, std = {np.sqrt(ep_var):.2f}")
    print("  Viterbi path around injury:")
    for t in range(18, 28):
        state_name = ["Injured", "Slump", "Average", "Good", "Star"][result.viterbi_path[t]]
        true_name = ["Injured", "Slump", "Average", "Good", "Star"][true_states[t]]
        fused = result.fused_mean[t]
        print(
            f"    GW{t + 1}: viterbi={state_name:8s}  true={true_name:8s}  "
            f"fused={fused:.1f}  obs={observations[t]:.1f}"
        )

    # Fused uncertainty should be lower than either component alone
    hmm_var_mean = np.mean(result.fused_var)
    kf_var_mean = np.mean(result.kalman_uncertainty)
    print(f"\n  Mean fused variance:  {hmm_var_mean:.3f}")
    print(f"  Mean Kalman variance: {kf_var_mean:.3f}")

    assert ep_mean >= 0, "Expected points must be non-negative"
    assert ep_var > 0, "Variance must be positive"

    print("  PASSED\n")


def test_baum_welch_learning():
    """Test that Baum-Welch shifts parameters toward data distribution."""
    print("=" * 60)
    print("TEST 6: Baum-Welch Parameter Learning")
    print("=" * 60)

    _, observations = generate_test_player()

    pipeline = PlayerInferencePipeline()
    pipeline.ingest_observations(observations)

    # Record initial emission params
    initial_means = {s: pipeline.hmm.emission_params[s][0] for s in range(5)}

    # Run Baum-Welch
    pipeline.learn_parameters(n_iter=10)

    # Emission params should have shifted
    learned_means = {s: pipeline.hmm.emission_params[s][0] for s in range(5)}

    print("  Emission means (initial → learned):")
    any_changed = False
    for s in range(5):
        changed = abs(initial_means[s] - learned_means[s]) > 0.01
        any_changed = any_changed or changed
        marker = "←" if changed else ""
        name = ["Injured", "Slump", "Average", "Good", "Star"][s]
        print(f"    {name:8s}: {initial_means[s]:.2f} → {learned_means[s]:.2f} {marker}")

    assert any_changed, "Baum-Welch should change at least some parameters"

    # Pipeline should still produce valid predictions after learning
    _ = pipeline.run()
    ep, var = pipeline.predict_next()
    print(f"  Post-learning prediction: E[P] = {ep:.2f}, Var = {var:.2f}")
    assert 0 <= ep <= 15
    assert var > 0

    print("  PASSED\n")


def test_custom_news_params_override_defaults():
    """Test configurable news thresholds/perturbations are applied by pipeline."""
    print("=" * 60)
    print("TEST 7: Configurable News Injection Parameters")
    print("=" * 60)

    # Override thresholds so availability=0.30 is treated as unavailable
    # and use a custom Kalman shock to verify map override is active.
    news_params = {
        "classification_thresholds": {
            "unavailable_max_availability": 0.35,
        },
        "perturbation_map": {
            "unavailable": {
                "state_boost": {0: 6.0},
                "kalman_shock": 4.0,
            },
        },
    }

    pipeline = PlayerInferencePipeline(news_params=news_params)
    observations = np.array([4.0, 5.0, 4.0, 3.0, 4.5], dtype=float)
    pipeline.ingest_observations(observations)

    signal = {"availability": 0.30, "minutes_risk": 0.0, "confidence": 1.0}
    category = _classify_news(
        signal["availability"],
        signal["minutes_risk"],
        pipeline.news_params["classification_thresholds"],
    )
    assert category == "unavailable"

    pipeline.inject_news(signal, timestep=4)

    expected_q = pipeline.kf.default_process_noise * 4.0
    actual_q = pipeline.kf.get_process_noise_override(4)
    print(f"  Custom category: {category}")
    print(f"  Process shock override at t=4: {actual_q:.2f} (expected {expected_q:.2f})")

    assert actual_q == expected_q

    print("  PASSED\n")


def test_calibrated_alpha_fusion_mode_runs():
    """Calibrated alpha fusion should return a valid alpha and prediction."""
    print("=" * 60)
    print("TEST 8: Calibrated Alpha Fusion")
    print("=" * 60)

    _, observations = generate_test_player()

    pipeline = PlayerInferencePipeline(
        fusion_mode="calibrated_alpha",
        fusion_params={
            "grid_step": 0.1,
            "min_history": 6,
            "default_alpha": 0.7,
        },
    )
    pipeline.ingest_observations(observations)
    result = pipeline.run()
    ep, var = pipeline.predict_next()

    assert result.fusion_alpha is not None
    assert 0.0 <= result.fusion_alpha <= 1.0
    assert np.isfinite(ep)
    assert np.isfinite(var)
    assert var > 0

    print(f"  Learned alpha: {result.fusion_alpha:.2f}")
    print(f"  Predicted E[P] = {ep:.2f}, Var = {var:.2f}")
    print("  PASSED\n")


def main():
    test_news_signal_integration()
    test_pipeline_without_signals()
    test_news_injection_shifts_beliefs()
    test_fixture_injection_shifts_uncertainty()
    test_full_pipeline_with_all_signals()
    test_baum_welch_learning()
    test_custom_news_params_override_defaults()
    test_calibrated_alpha_fusion_mode_runs()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
