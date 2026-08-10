# MSK module — managed Kafka broker.
#
# Design notes (research-backed):
# - MSK is the AWS-managed Kafka-protocol broker. Redpanda is a drop-in,
#   Kafka-API-compatible alternative (Redpanda/Amazon: MSF consumes it via the
#   native Kafka connector); the Python producers/consumers in this repo speak
#   the Kafka API either way, so the broker is swappable behind the same code.
# - Provisioned MSK for a sustained workload; MSK Serverless for bursty.
#   Dev defaults are kept small, but the cluster is always 3 brokers across the
#   AZs — replication.factor 3 / min.insync.replicas 2 is the Kafka durability
#   floor, and "one broker" is not a real Kafka cluster (factualminds tiering:
#   MSK only when the Kafka protocol is actually needed; a 3-broker cluster is
#   the smallest honest deployment).
# - client_broker = TLS_PLAINTEXT keeps dev wiring simple; production moves to
#   TLS + IAM SASL (aws-msk-iam-auth) so no long-lived credentials exist at all
#   (AWS Managed Flink guidance: role-based auth, never baked-in secrets).
# - encrypted-in-transit and at-rest by default; enhanced_monitoring gives
#   per-broker metrics the CloudWatch alarms in the observability module feed on.

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })
}

resource "aws_msk_configuration" "this" {
  name           = "${local.name}-config"
  kafka_versions = [var.kafka_version]

  server_properties = <<-EOT
    auto.create.topics.enable = true
    default.replication.factor = 3
    min.insync.replicas = 2
    num.partitions = 1
    log.retention.hours = 168
  EOT
}

resource "aws_msk_cluster" "this" {
  cluster_name           = "${local.name}-msk"
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.broker_count

  configuration_info {
    arn      = aws_msk_configuration.this.arn
    revision = aws_msk_configuration.this.latest_revision
  }

  broker_node_group_info {
    instance_type   = var.broker_instance_type
    client_subnets  = var.private_subnet_ids
    security_groups = [var.internal_security_group_id]
    az_distribution = "DEFAULT"

    storage_info {
      ebs_storage_info {
        volume_size = var.broker_storage_gb
      }
    }
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS_PLAINTEXT"
      in_cluster    = true
    }
  }

  enhanced_monitoring = "PER_BROKER"

  tags = local.tags
}
