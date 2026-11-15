FROM python:3.12-slim
WORKDIR /app
COPY requirements-base.txt requirements-base.txt
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-base.txt -r requirements.txt
COPY app ./app
COPY run.py ./run.py
RUN useradd --create-home appuser && mkdir -p data/uploads data/chroma_db && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["python", "run.py"]
