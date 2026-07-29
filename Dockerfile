# Two stages. The builder has uv and resolves the environment; the runtime
# image receives that environment and nothing else — no uv, no lockfile, no
# source tree, no package manager left inside to run.

FROM ghcr.io/astral-sh/uv:0.5.30-python3.13-bookworm-slim AS builder

# copy: the venv is COPYed into another image, where uv's default hardlinks
#   back into its cache would dangle.
# bytecode: compile here, so the runtime never writes .pyc into a directory it
#   deliberately does not own.
# downloads=never: use this image's Python, so the runtime stage's interpreter
#   is the one the venv was built against.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies alone, from the lockfile alone. This layer is most of the image
# — spaCy's en_core_web_lg is 433MB of it — and must not rebuild when a node
# changes. --no-dev keeps pytest and deepeval out of production.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --no-editable installs the project as a built wheel, so what runs in the
# container is exactly the `packages` list in pyproject.toml. An editable
# install puts /app on sys.path and hides a package missing from that list.
COPY accounts accounts
COPY app app
COPY core core
COPY kb kb
COPY observability observability
COPY pii pii
COPY rag rag
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim-bookworm

# Unbuffered is load-bearing, not hygiene: the per-request trace is a required
# deliverable written to stdout, and stdout is a pipe here. Buffered, a
# container that dies loses exactly the traces that would explain why.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

# The base image ships pip and nothing here uses it — the app runs from the
# venv, which uv builds without one. Removing it denies a foothold the means to
# fetch tooling, and turns Presidio's fallback of pip-installing a missing
# spaCy model into a loud failure instead of a network call on the request path.
# Absolute path because PATH above already points at the venv.
RUN /usr/local/bin/python -m pip uninstall --yes pip

# Owned by root, run as someone else: the process cannot rewrite its own code.
# Same path as the builder, because the venv's scripts hardcode their shebang.
RUN useradd --system --no-create-home --uid 10001 mal
COPY --from=builder --chown=root:root /app/.venv /app/.venv

USER mal
EXPOSE 8000

# sh -c to expand the $PORT Railway injects; exec so uvicorn replaces the shell
# and receives SIGTERM itself instead of having it swallowed by a PID 1 that
# does not forward signals.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
