# Contributing

Controller submissions follow [docs/SUBMISSION_GUIDE.md](docs/SUBMISSION_GUIDE.md). Engine changes should include focused tests, preserve replay/version compatibility or increment the appropriate version constant, and keep controller-facing APIs minimal.

Install development dependencies and run:

```bash
python -m pip install -e ".[dev,competition,render]"
python -m pytest
```

Do not combine engine changes with a community controller submission. Security-boundary changes should explain which jobs execute untrusted code and which credentials they receive.

