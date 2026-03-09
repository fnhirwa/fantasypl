# Installation

## From Source

```bash
git clone https://github.com/fnhirwa/fantasypl.git
cd fantasypl
pip install -e "."
```

## Optional Dependencies

=== "ML Models"

    Adds Ridge, XGBoost, LightGBM regression models.

    ```bash
    pip install -e ".[ml]"
    ```

=== "Optimization"

    Adds PuLP for ILP-based optimal squad selection.

    ```bash
    pip install -e ".[optimization]"
    ```

=== "Development"

    Adds pytest, ruff, mypy, pre-commit.

    ```bash
    pip install -e ".[dev]"
    ```

=== "Everything"

    ```bash
    pip install -e ".[all]"
    ```

## Requirements

- Python 3.9+
- NumPy, Pandas, Requests, Pydantic (installed automatically)
- SciPy (used by the inference pipeline)
