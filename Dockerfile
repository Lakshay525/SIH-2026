# Reproducible run: `docker compose up` gets a judge from clone to a live
# dashboard without depending on whatever Python/OS they happen to have.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit's own healthcheck endpoint -- lets `docker compose up` report
# "healthy" only once the dashboard is actually serving, not just started.
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
