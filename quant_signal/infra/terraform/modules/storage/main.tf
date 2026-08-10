# Storage module — S3 (Flink checkpoints, research artifacts) + ElastiCache Redis.
#
# S3 roles, research-backed:
# - Flink checkpoints: RocksDB state snapshots land in S3 so a task restart
#   (or an AZ loss) restores from the last good checkpoint instead of replaying
#   the whole topic. Checkpointing must stay enabled in production (AWS Managed
#   Flink guidance); 2-5 min intervals are the balanced default for most jobs.
# - Research artifacts: MLflow runs / harness leaderboards / swept configs are
#   immutable objects — versioning + an IA transition keeps a durable audit
#   trail without a hot tier. Nothing that matters is ever force-destroyed.
# - ElastiCache Redis is the online store (same role as the local Redis :6380):
#   bounded feature lists, sub-500ms API reads. Dev uses a single node;
#   production upgrades to Redis Cluster mode (multi-shard) + Multi-AZ.

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })
}

resource "aws_s3_bucket" "checkpoints" {
  bucket        = var.checkpoint_bucket_name
  force_destroy = false
  tags          = merge(local.tags, { Name = "${local.name}-checkpoints" })
}

resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"

    filter {}

    expiration {
      days = var.checkpoint_retention_days
    }
  }
}

resource "aws_s3_bucket" "research" {
  bucket        = var.research_bucket_name
  force_destroy = false
  tags          = merge(local.tags, { Name = "${local.name}-research" })
}

resource "aws_s3_bucket_versioning" "research" {
  bucket = aws_s3_bucket.research.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "research" {
  bucket = aws_s3_bucket.research.id

  rule {
    id     = "archive-to-ia"
    status = "Enabled"

    filter {}

    transition {
      days          = var.research_ia_days
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = var.research_expiration_days
    }
  }
}

resource "aws_elasticache_subnet_group" "this" {
  name       = "${local.name}-redis"
  subnet_ids = var.private_subnet_ids
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id               = "${local.name}-redis"
  engine                   = "redis"
  engine_version           = var.redis_engine_version
  node_type                = var.redis_node_type
  num_cache_nodes          = 1
  parameter_group_name     = var.redis_parameter_group
  port                     = 6379
  subnet_group_name        = aws_elasticache_subnet_group.this.name
  security_group_ids       = [var.internal_security_group_id]
  snapshot_retention_limit = 0
  apply_immediately        = true
  tags                     = local.tags
}
