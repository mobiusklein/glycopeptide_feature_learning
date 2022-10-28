dev:
	pip install -v -e . --no-build-isolation


test:
	pytest -v -s tests --cov=glycopeptide_feature_learning --cov-report=html --cov-report term