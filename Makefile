.PHONY: setup data train app clean help

help:
	@echo "Clinical Readmission Risk — NLP Pipeline"
	@echo ""
	@echo "Commands:"
	@echo "  make setup    Install Python dependencies"
	@echo "  make data     Download MTSamples dataset"
	@echo "  make train    Train the classifier (XGBoost by default)"
	@echo "  make app      Launch Streamlit demo"
	@echo "  make notebook Launch Jupyter notebooks"
	@echo "  make clean    Remove cached model artifacts"

setup:
	pip install -r requirements.txt

data:
	python data/fetch_data.py

train:
	python src/train.py --model xgboost

train-logistic:
	python src/train.py --model logistic

app:
	streamlit run app/streamlit_app.py

notebook:
	jupyter notebook notebooks/

clean:
	rm -f models/embeddings_cache.npy
	rm -f models/classifier.joblib
	rm -f models/pca.joblib
	rm -f models/train_metadata.json
	@echo "Cleaned model artifacts. Data and results are preserved."

clean-all: clean
	rm -f data/mtsamples.csv
	rm -f results/*.png results/*.json
	@echo "Full clean complete."
