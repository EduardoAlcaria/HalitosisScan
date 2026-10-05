FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"

COPY hality/ hality/
COPY models/tonguenet.pt models/classificador.pkl models/gate.pkl models/segmentador.pt models/

ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "hality.api:app", "--host", "0.0.0.0", "--port", "8000"]
