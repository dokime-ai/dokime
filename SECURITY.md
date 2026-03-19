# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Dokime, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email **security@dokime.ai** with:

1. A description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact assessment
4. Any suggested fixes (optional)

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days for critical issues.

## Scope

### What Dokime does

Dokime is a **local data processing tool**. It:

- Reads datasets from local files or HuggingFace Hub
- Processes data locally on your machine
- Writes output to local files or HuggingFace Hub
- Runs a local web server for the `explore` command (bound to localhost by default)

### Data handling

- Dokime does **not** transmit your data to any external service (unless you explicitly push to HuggingFace Hub)
- The `explore` web UI binds to `127.0.0.1` by default. If you change `--host` to `0.0.0.0`, the server becomes network-accessible — use this only in trusted environments
- Embedding models are downloaded from HuggingFace Hub and cached locally

### Dependencies

Dokime depends on well-established open-source libraries (PyArrow, scikit-learn, sentence-transformers, FastAPI). We monitor dependencies for known vulnerabilities and update promptly.

### Code execution

Dokime does **not** execute arbitrary code from datasets. YAML pipeline configs are parsed with `yaml.safe_load()` which prevents code injection. Custom filters require explicit Python code — they are not loaded from untrusted sources.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | Best effort |

## Security Best Practices for Users

1. **Review datasets before processing** — Dokime processes data as-is; malformed data may cause unexpected behavior
2. **Keep Dokime updated** — run `pip install --upgrade dokime` regularly
3. **Don't expose the explore server** — keep `--host 127.0.0.1` (the default) unless you have a specific reason to change it
4. **Use virtual environments** — isolate Dokime's dependencies from your system Python
