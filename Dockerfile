FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

COPY model/ model/

COPY src/api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "localhost", "--port", "8000"]