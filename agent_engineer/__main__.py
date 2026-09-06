"""Top-level executable entry point for python -m agent_engineer."""

from __future__ import annotations

import sys

from agent_engineer.cli import main

if __name__ == "__main__":
    sys.exit(main())
