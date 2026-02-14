from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> bool:
	"""Load environment variables from a .env file.

	- Minimal parser: KEY=VALUE, ignores blank lines and comments.
	- Does not override existing env vars unless override=True.
	- Returns True if a file was found and parsed.

	This is intentionally tiny to keep py-mono dependency-light.
	"""
	p = Path(path)
	if not p.is_file():
		return False

	for raw_line in p.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key:
			continue

		# Strip simple quotes
		if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
			value = value[1:-1]

		if not override and key in os.environ:
			continue
		os.environ[key] = value

	return True
