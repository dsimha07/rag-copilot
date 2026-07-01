# Makefile — dev shortcuts. Placeholder (echoes intended commands only).
.PHONY: ingest eval api dashboard up

ingest:
	@echo "TODO: python -m src.ingest.cli --source data/corpus --rebuild"

eval:
	@echo "TODO: python -m src.eval.run --strategy hybrid"

api:
	@echo "TODO: uvicorn src.api.app:app --reload"

dashboard:
	@echo "TODO: streamlit run dashboards/copilot_app.py"

up:
	@echo "TODO: docker compose up"
