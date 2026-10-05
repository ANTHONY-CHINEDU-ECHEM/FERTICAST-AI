FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY app ./app
COPY .streamlit ./.streamlit
COPY models ./models

RUN useradd --create-home ferticast && chown -R ferticast /app
USER ferticast
EXPOSE 8000 8501
CMD ["uvicorn", "ferticast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
