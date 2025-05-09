FROM python:3.10

WORKDIR /app

RUN useradd -m -r appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/chroma_db /app/offload && \
    chown -R appuser:appuser /app

COPY . /app

USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]