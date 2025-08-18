FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY goa_nonexp.py .
COPY retrieve_terms.py .
COPY ontology.py .
COPY config.yaml .

ENTRYPOINT ["python3", "goa_nonexp.py"]