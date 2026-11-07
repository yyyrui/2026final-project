#!/usr/bin/env python3
"""Compare three models on every sales series; one joint TACTiS-2, no grouping or bagging."""
from comparison.runner import main


if __name__ == "__main__":
    main(mode="all")
