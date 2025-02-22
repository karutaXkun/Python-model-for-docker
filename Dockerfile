FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "Multiple.py"]