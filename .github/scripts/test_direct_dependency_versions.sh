#!/usr/bin/env bash
set -euo pipefail

resolution="${1:-}"
case "$resolution" in
  lowest-direct | highest) ;;
  *)
    echo "Usage: $0 {lowest-direct|highest}" >&2
    exit 2
    ;;
esac

: "${PKG_NAME:?PKG_NAME must be set}"
: "${PYTHON_VERSION:?PYTHON_VERSION must be set}"
: "${RUNNER_TEMP:?RUNNER_TEMP must be set}"

export UV_NO_SYNC=1

requirements_file="$RUNNER_TEMP/$resolution-direct-dependencies.txt"

uv pip compile pyproject.toml \
  --no-deps \
  --no-sources \
  --python-version "$PYTHON_VERSION" \
  --resolution "$resolution" \
  --output-file "$requirements_file"

# Test the LangChain dependency family separately once its lower bounds are
# mutually compatible. Installing with --no-deps preserves its dependency tree.
awk '!/^(langchain|langgraph|orjson==|pydantic==|requests==|typing-extensions==)/' \
  "$requirements_file" > "$requirements_file.filtered"
mv "$requirements_file.filtered" "$requirements_file"

if ! grep -qE '^[[:alnum:]]' "$requirements_file"; then
  echo "No direct dependencies remain after filtering deferred packages." >&2
  exit 1
fi

echo "Testing direct dependencies resolved with: $resolution"
cat "$requirements_file"

uv pip install --no-deps --requirements "$requirements_file"

if [[ "$PKG_NAME" == "langchain-azure-postgresql" ]]; then
  echo "Running unit and integration tests"
  uv run --no-sync pytest
else
  echo "Running unit tests"
  make tests
  
  echo "Running integration tests"
  make integration_tests
fi