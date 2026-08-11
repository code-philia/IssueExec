#!/usr/bin/env python3
"""Compatibility entry point for the IssueExec localization CLI."""

from issueexec.cli import *  # noqa: F401,F403
from issueexec.cli import main


if __name__ == "__main__":
    main()
