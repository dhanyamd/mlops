# Observability module — CloudWatch alarms + dashboard for the platform.
#
# What gets watched and why (research-backed):
# - MSK `UnderReplicatedPartitions`: the single most important Kafka health
#   signal — > 0 means the cluster is degraded and a broker loss will hurt.
# - MSK broker CPU (sum across brokers): brokers burning > 80% sustained are
#   one rebalance away from a latency cliff (Kafka deviates hard under CPU
#   contention — clients get throttled before the broker falls over).
# - MSK `KafkaDataLogsDiskUsed`: disks at 85%+ force partition migration.
# - ElastiCache `DatabaseMemoryUsagePercentage`: the online store is an
#   in-memory cache; an eviction/swap spiral shows up here first.
# - ALB `HTTPCode_Target_5XX_Count`: the UI front door — 5xx = the API is
#   breaking for users, not just internally.
# - ECS `CPUUtilization`/`MemoryUtilization` per service, plus a "no running
#   tasks" alarm (sum == 0) that fires when a Fargate service silently stops.
#
# Alarms fan into one SNS topic (dev keeps it lightweight; prod can add PagerDuty
# as a second subscription without touching the alarms).

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })

  # MSK exposes per-broker CPU under a 1-indexed "Broker ID" dimension.
  msk_broker_ids = [for i in range(var.msk_broker_count) : tostring(i + 1)]

  # CPU alarm thresholds: sum-of-brokers must stay under broker_count * pct.
  msk_cpu_threshold = var.msk_cpu_threshold_pct * var.msk_broker_count
}

resource "aws_sns_topic" "this" {
  name = "${local.name}-alerts"
  tags = local.tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alerts_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.this.arn
  protocol  = "email"
  endpoint  = var.alerts_email
}

# ── MSK ──────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "msk_under_replicated_partitions" {
  alarm_name          = "${local.name}-msk-under-replicated-partitions"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = 0
  statistic           = "Maximum"
  metric_name         = "UnderReplicatedPartitions"
  namespace           = "AWS/Kafka"
  dimensions = {
    "Cluster Name" = var.msk_cluster_name
  }
  alarm_description  = "MSK has under-replicated partitions — replication is degraded"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

resource "aws_cloudwatch_metric_alarm" "msk_broker_cpu" {
  alarm_name          = "${local.name}-msk-broker-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = local.msk_cpu_threshold
  alarm_description   = "Aggregate MSK broker CPU above ${var.msk_cpu_threshold_pct}%"
  alarm_actions       = [aws_sns_topic.this.arn]
  ok_actions          = [aws_sns_topic.this.arn]
  treat_missing_data  = "notBreaching"
  tags                = local.tags

  metric_query {
    id         = "sum_cpu"
    expression = join(" + ", [for id in local.msk_broker_ids : "m${id}"])
    label      = "MSK broker CPU sum (%)"
  }

  dynamic "metric_query" {
    for_each = local.msk_broker_ids
    content {
      id = "m${metric_query.value}"
      metric {
        namespace   = "AWS/Kafka"
        metric_name = "CpuUser"
        period      = 300
        stat        = "Average"
        dimensions = {
          "Cluster Name" = var.msk_cluster_name
          "Broker ID"    = metric_query.value
        }
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "msk_disk_usage" {
  alarm_name          = "${local.name}-msk-disk-usage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.msk_disk_threshold_pct
  statistic           = "Maximum"
  metric_name         = "KafkaDataLogsDiskUsed"
  namespace           = "AWS/Kafka"
  dimensions = {
    "Cluster Name" = var.msk_cluster_name
  }
  alarm_description  = "MSK data-logs disk above ${var.msk_disk_threshold_pct}%"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

# ── ElastiCache Redis (online store) ─────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name          = "${local.name}-redis-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.redis_memory_threshold_pct
  statistic           = "Average"
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  dimensions = {
    CacheClusterId = var.redis_cluster_id
  }
  alarm_description  = "Redis (online store) memory above ${var.redis_memory_threshold_pct}%"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name          = "${local.name}-redis-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.redis_cpu_threshold_pct
  statistic           = "Average"
  metric_name         = "EngineCPUUtilization"
  namespace           = "AWS/ElastiCache"
  dimensions = {
    CacheClusterId = var.redis_cluster_id
  }
  alarm_description  = "Redis (online store) engine CPU above ${var.redis_cpu_threshold_pct}%"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

# ── ALB (UI front door) ──────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${local.name}-alb-target-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.alb_5xx_threshold
  statistic           = "Sum"
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  dimensions = {
    LoadBalancer = var.alb_name
  }
  alarm_description  = "ALB target 5xx above ${var.alb_5xx_threshold} per 5m window"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

# ── ECS services (one alarm set per service) ─────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "ecs_cpu" {
  for_each            = toset(var.service_names)
  alarm_name          = "${local.name}-ecs-cpu-${each.key}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.ecs_cpu_threshold_pct
  statistic           = "Average"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  dimensions = {
    ClusterName = var.cluster_name
    ServiceName = each.key
  }
  alarm_description  = "ECS service ${each.key} CPU above ${var.ecs_cpu_threshold_pct}%"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

resource "aws_cloudwatch_metric_alarm" "ecs_memory" {
  for_each            = toset(var.service_names)
  alarm_name          = "${local.name}-ecs-memory-${each.key}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = var.ecs_memory_threshold_pct
  statistic           = "Average"
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  dimensions = {
    ClusterName = var.cluster_name
    ServiceName = each.key
  }
  alarm_description  = "ECS service ${each.key} memory above ${var.ecs_memory_threshold_pct}%"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "notBreaching"
  tags               = local.tags
}

# Sum == 0 over a full evaluation window means no task is running — Fargate
# services don't emit "down", they stop emitting. evaluate_low_sample_count
# forces the alarm to honor the zero instead of treating it as missing.
resource "aws_cloudwatch_metric_alarm" "ecs_no_running_tasks" {
  for_each            = toset(var.service_names)
  alarm_name          = "${local.name}-ecs-down-${each.key}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  period              = 300
  threshold           = 1
  statistic           = "Sum"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  dimensions = {
    ClusterName = var.cluster_name
    ServiceName = each.key
  }
  alarm_description  = "ECS service ${each.key} has no running tasks"
  alarm_actions      = [aws_sns_topic.this.arn]
  ok_actions         = [aws_sns_topic.this.arn]
  treat_missing_data = "breaching"
  tags               = local.tags
}
