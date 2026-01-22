"""Basic usage example for FPLX."""

from fplx import FPLModel


def main():
    """Basic squad selection example."""
    print("Initializing FPLX model...")
    model = FPLModel(budget=100.0, horizon=1, formation="auto")

    # Load data from FPL API
    print("Loading player data from FPL API...")
    model.load_data(source="api")

    # Fit the model
    print("Fitting model and generating predictions...")
    model.fit()

    # Select best 11 players
    print("Optimizing squad selection...")
    squad = model.select_best_11()

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMAL SQUAD")
    print("=" * 60)
    print(f"Formation: {squad.formation}")
    print(f"Total Cost: £{squad.total_cost:.1f}m")
    print(f"Expected Points: {squad.expected_points:.2f}")
    print(f"Captain: {squad.captain.name if squad.captain else 'None'}")
    print("\nPlayers:")
    print(squad.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
