FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MOCK_LLM=1
COPY support_assistant/requirements.txt /app/support_assistant/requirements.txt
RUN pip install --no-cache-dir -r /app/support_assistant/requirements.txt
COPY support_assistant /app/support_assistant
RUN python -m pip install --no-cache-dir --upgrade pip
EXPOSE 7860
CMD ["sh", "-c", "python support_assistant/ingest.py && uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860"]
