.PHONY: install data train report all test app api docker clean

install:
	pip install -e ".[dev]"

data:
	python -m ferticast.cli simulate

train:
	python -m ferticast.cli train

report:
	python -m ferticast.cli report

all:
	python -m ferticast.cli all

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn ferticast.api.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

clean:
	rm -rf data/raw/*.csv data/processed models/*.joblib reports/*.gz .pytest_cache
