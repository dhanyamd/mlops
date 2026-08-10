output "sns_topic_arn" {
  description = "ARN of the alerts topic (for pager/chat integrations)"
  value       = aws_sns_topic.this.arn
}

output "alarm_names" {
  description = "All alarm names for quick `aws cloudwatch` lookups"
  value = concat(
    [
      aws_cloudwatch_metric_alarm.msk_under_replicated_partitions.alarm_name,
      aws_cloudwatch_metric_alarm.msk_broker_cpu.alarm_name,
      aws_cloudwatch_metric_alarm.msk_disk_usage.alarm_name,
      aws_cloudwatch_metric_alarm.redis_memory.alarm_name,
      aws_cloudwatch_metric_alarm.redis_cpu.alarm_name,
      aws_cloudwatch_metric_alarm.alb_5xx.alarm_name,
    ],
    [for k, v in aws_cloudwatch_metric_alarm.ecs_cpu : v.alarm_name],
  )
}
