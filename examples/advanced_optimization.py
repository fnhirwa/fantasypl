"""Advanced optimization example with custom configuration."""

from fplx import FPLModel

def main():
    """Advanced squad selection with ILP optimizer."""
    config = {
        'model_type': 'form_based',
        'optimizer': 'ilp',
        'feature_engineering': {
            'rolling_windows': [3, 5, 10],
            'lag_periods': [1, 2, 3, 7],
            'ewma_alphas': [0.2, 0.3, 0.5],
        },
        'signals': {
            'stats_weights': {
                'points_mean': 0.35,
                'xG_mean': 0.15,
                'xA_mean': 0.15,
                'minutes_consistency': 0.20,
                'form_trend': 0.15,
            },
        },
    }
    
    print("Initializing FPLX with custom configuration...")
    model = FPLModel(
        budget=100.0,
        horizon=3,
        formation="3-4-3",
        config=config
    )
    
    # Load data
    print("Loading data...")
    model.load_data(source='api')
    
    # Optional: load news data
    # model.load_news('news.json')
    
    print("Training model...")
    model.fit()
    
    # Optimize squad with ILP (optimal solution)
    print("Running ILP optimization (this may take a minute)...")
    squad = model.select_best_11()
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMAL SQUAD (ILP Optimization)")
    print("="*70)
    print(f"Formation: {squad.formation}")
    print(f"Total Cost: £{squad.total_cost:.1f}m")
    print(f"Expected Points (next 3 GWs): {squad.expected_points:.2f}")
    print(f"Captain: {squad.captain.name if squad.captain else 'None'}")
    print(f"\nConstraints satisfied: {squad.validate_formation()}")
    
    print("\nPlayers:")
    print(squad.summary())
    print("="*70)


if __name__ == "__main__":
    main()
