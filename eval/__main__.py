"""Entry point for "`make eval`"."""

from __future__ import annotations

import sys

from eval.run import main

if __name__ == "__main__":
    sys.exit(main())
