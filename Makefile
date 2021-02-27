dev:
	python setup.py develop


test:
	pytest -v -s glycopeptide_feature_learning --cov=glycopeptide_feature_learning --cov-report=html --cov-report term