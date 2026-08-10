FROM flink:1.19.3-scala_2.12

# Kafka SQL connector (shaded) for the crypto_features job. The base image runs
# as the `flink` user, so the ADD-ed jar must be owned/readable by it — otherwise
# the planner fails with ClassNotFoundException when CREATE TABLE is validated.
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.19/flink-sql-connector-kafka-3.2.0-1.19.jar /opt/flink/lib/
RUN chown flink:flink /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar \
    && chmod 644 /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar

# S3 filesystem plugin so RocksDB checkpoints can be written to s3a:// (the
# AWS/Fargate deployment stores state in S3; local compose never uses the
# scheme). MUST go under plugins/, not lib/: Flink ships these jars with
# relocations removed, so placing them in lib/ breaks the JVM
# (nightlies.apache.org/flink/flink-docs-release-1.19/docs/deployment/filesystems/s3).
# The task role's default credential chain (Fargate container credentials)
# authorizes the S3 calls — no keys in the image.
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-s3-fs-hadoop/1.19.3/flink-s3-fs-hadoop-1.19.3.jar /opt/flink/plugins/s3-fs-hadoop/
RUN chown -R flink:flink /opt/flink/plugins/s3-fs-hadoop \
    && chmod 644 /opt/flink/plugins/s3-fs-hadoop/flink-s3-fs-hadoop-1.19.3.jar

RUN mkdir -p /opt/flink/jobs
# 5m job = benchmark/Alpha-Autopsy book; 1h job = live trading clock.
COPY jobs/crypto_features.sql /opt/flink/jobs/
COPY jobs/crypto_features_1h.sql /opt/flink/jobs/

# Checkpoint store must be writable by the `flink` user (fresh named volumes
# inherit this directory's ownership when first mounted).
RUN mkdir -p /opt/flink/checkpoints && chown -R flink:flink /opt/flink/checkpoints
