#!/usr/bin/env python3
"""Compare TACTiS2, Chronos-2 Small and M4 Naive on a fixed sample of 5,000 complete sales series."""
from comparison.runner import main


if __name__ == "__main__":
    main(mode="sample")
