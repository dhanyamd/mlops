# CloudWatch dashboard — a single pane for the platform's vital signs.
# Same metrics the alarms watch, so the page the on-call opens matches the
# pages they get. Dashboard JSON is generated with jsonencode (never a
# hand-pasted blob) so it stays in sync with the module's inputs.

resource "aws_cloudwatch_dashboard" "platform" {
  dashboard_name = "${local.name}-platform"
  dashboard_body = jsonencode({
    start          = "-PT6H"
    periodOverride = "inherit"
    widgets = concat(
      [
        {
          type   = "text"
          x      = 0
          y      = 0
          width  = 24
          height = 2
          properties = {
            markdown = "## ${local.name} platform\nMSK · Redis online store · ALB · ECS services. Alarms fan into the alerts SNS topic."
          }
        },
        {
          type   = "metric"
          x      = 0
          y      = 2
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              ["AWS/Kafka", "UnderReplicatedPartitions", { stat = "Max" }],
            ]
            region = var.region
            title  = "MSK — under-replicated partitions"
          }
        },
        {
          type   = "metric"
          x      = 12
          y      = 2
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              ["AWS/Kafka", "KafkaDataLogsDiskUsed", { stat = "Max" }],
            ]
            region = var.region
            title  = "MSK — data-logs disk used"
          }
        },
        {
          type   = "metric"
          x      = 0
          y      = 8
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              ["AWS/ElastiCache", "DatabaseMemoryUsagePercentage", "CacheClusterId", var.redis_cluster_id, { stat = "Average" }],
              [".", "EngineCPUUtilization", ".", ".", { stat = "Average", yAxis = "right" }],
            ]
            region = var.region
            title  = "Redis online store — memory / CPU"
          }
        },
        {
          type   = "metric"
          x      = 12
          y      = 8
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", var.alb_name, { stat = "Sum" }],
            ]
            region = var.region
            title  = "ALB — target 5xx"
          }
        },
      ],
      [
        # ECS per-service CPU (top) and memory (bottom) as one widget each.
        {
          type   = "metric"
          x      = 0
          y      = 14
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              for svc in var.service_names : [
                "AWS/ECS", "CPUUtilization",
                "ClusterName", var.cluster_name,
                "ServiceName", svc,
                { stat = "Average", label = svc },
              ]
            ]
            region = var.region
            title  = "ECS — service CPU %"
          }
        },
        {
          type   = "metric"
          x      = 12
          y      = 14
          width  = 12
          height = 6
          properties = {
            view    = "timeSeries"
            stacked = false
            metrics = [
              for svc in var.service_names : [
                "AWS/ECS", "MemoryUtilization",
                "ClusterName", var.cluster_name,
                "ServiceName", svc,
                { stat = "Average", label = svc },
              ]
            ]
            region = var.region
            title  = "ECS — service memory %"
          }
        },
      ],
    )
  })
}
