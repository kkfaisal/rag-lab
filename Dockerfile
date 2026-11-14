# rag-lab in one image: the built web UI plus the Python backend.

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
# git: run records store the commit, and the EnterpriseRAG-Bench judge is cloned with it.
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    RAG_LAB_HOME=/app HF_HOME=/hf-cache
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ backend/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev
COPY --from=web /web/dist web/dist
EXPOSE 8000
# Inside the container listen on all interfaces; docker-compose publishes the port on 127.0.0.1 only.
CMD ["rag-lab", "serve", "--host", "0.0.0.0", "--port", "8000"]
