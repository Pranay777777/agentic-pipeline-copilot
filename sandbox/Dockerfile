# syntax=docker/dockerfile:1
# The sandbox that runs generated notebooks (ADR-005). Built locally and in CI:
#   python -m copilot.sandbox build
# It is only ever started with no network, a read-only root filesystem, no
# capabilities and CPU/memory/process/time limits (copilot.sandbox.runner).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sfn "/usr/lib/jvm/$(ls /usr/lib/jvm | grep java-17 | head -1)" /usr/lib/jvm/java-17

ENV JAVA_HOME=/usr/lib/jvm/java-17

# Spark 4.0 and the Delta release built for it; pinned so a run is reproducible.
RUN pip install "pyspark==4.0.4" "delta-spark==4.0.1"

# Delta's jars are fetched once, here, so runs need no network.
COPY warm_delta.py /opt/sandbox/warm_delta.py
RUN python /opt/sandbox/warm_delta.py \
    && ls /opt/sandbox/jars/*.jar \
    && rm -rf /root/.ivy2 /tmp/*

# The static gate and the generated tests (ADR-006) run in here too, so the
# generated code is never linted, type-checked or tested on the host.
RUN pip install "ruff==0.16.9" "mypy==2.3.1" "pytest==9.1.1"

RUN useradd --uid 1000 --create-home sandbox
USER 1000
WORKDIR /tmp
CMD ["python", "/job/harness.py"]
