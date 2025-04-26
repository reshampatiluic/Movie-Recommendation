# Dockerfile

FROM python:3.11

WORKDIR /Movie-Recommendation

COPY . /Movie-Recommendation

# Inject the git commit hash as a build argument
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT_HASH=$GIT_COMMIT

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8082

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8082", "--reload"]