"""Example using ML models for predictions."""

from fplx import FPLModel


def main():
    """Example using XGBoost for predictions."""
    print("Initializing FPLX with XGBoost model...")
    config = {
        "model_type": "xgboost",
        "model_config": {
            "n_estimators": 100,
            "max_depth": 3,
        },
        "optimizer": "greedy",
    }
    model = FPLModel(budget=100.0, config=config)

    # Load data
    print("Loading data...")
    model.load_data(source="api")

    # Enrich with historical data (for ML models)
    print("Enriching player history...")
    model.players = model.data_loader.enrich_player_history(model.players)

    # fit model
    print("Training XGBoost model...")
    model.fit()

    # select squad
    print("Selecting squad...")
    squad = model.select_best_11()

    # results
    print("\n" + "=" * 60)
    print("SQUAD (ML-based predictions)")
    print("=" * 60)
    print(squad.summary())
    print("=" * 60)


if __name__ == "__main__":
    # Note: Requires fplx[ml] installation
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install with ML support: pip install fplx[ml]")
