FROM flink:1.19.3-scala_2.12

# Kafka SQL connector (shaded) for the crypto_features job. The base image runs
# as the `flink` user, so the ADD-ed jar must be owned/readable by it — otherwise
# the planner fails with ClassNotFoundException when CREATE TABLE is validated.
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.19/flink-sql-connector-kafka-3.2.0-1.19.jar /opt/flink/lib/
RUN chown flink:flink /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar \
    && chmod 644 /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar

RUN mkdir -p /opt/flink/jobs
COPY jobs/crypto_features.sql /opt/flink/jobs/

# Checkpoint store must be writable by the `flink` user (fresh named volumes
# inherit this directory's ownership when first mounted).
RUN mkdir -p /opt/flink/checkpoints && chown -R flink:flink /opt/flink/checkpoints
