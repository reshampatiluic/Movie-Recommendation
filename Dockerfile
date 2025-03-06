FROM python:3.11

WORKDIR /Movie-Recommendation

COPY . /Movie-Recommendation

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8082

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8082", "--reload"]