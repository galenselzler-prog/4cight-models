# 4Cight labeling: Label Studio Community (Apache-2.0), pinned.
# Data lives in Render Postgres; audio stays in the private R2 research
# bucket and is shown to labelers through short-lived links.
FROM python:3.12-slim
RUN pip install --no-cache-dir "label-studio==1.23.1"
# Blank Sentry DSNs: no error reports (which can contain task data) are sent
# to Label Studio's developers.
ENV DJANGO_DB=default \
    LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true \
    SENTRY_DSN="" \
    FRONTEND_SENTRY_DSN="" \
    LOG_LEVEL=WARNING
EXPOSE 8080
CMD ["sh", "-c", "label-studio start --no-browser --host 0.0.0.0 --port ${PORT:-8080}"]
