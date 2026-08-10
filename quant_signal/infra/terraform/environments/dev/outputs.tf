output "vpc_id" {
  description = "Platform VPC"
  value       = module.networking.vpc_id
}

output "alb_dns_name" {
  description = "UI front door (http://<dns> after deployment)"
  value       = module.ecs.alb_dns_name
}

output "ecs_cluster_name" {
  description = "Cluster name for `aws ecs` / ecs exec"
  value       = module.ecs.cluster_name
}

output "msk_bootstrap_brokers" {
  description = "Kafka bootstrap endpoints for the app agents (plaintext)"
  value       = module.msk.bootstrap_brokers_plaintext
}

output "redis_endpoint" {
  description = "Redis online-store endpoint"
  value       = "${module.storage.redis_endpoint_address}:${module.storage.redis_port}"
}

output "checkpoint_bucket" {
  description = "Flink checkpoint bucket (s3a:// prefix used in task defs)"
  value       = module.storage.checkpoint_bucket_name
}

output "ecr_repository_urls" {
  description = "Registry URLs for the app/flink/ui images"
  value       = module.ecr.repository_urls
}

output "alerts_sns_topic_arn" {
  description = "SNS topic CloudWatch alarms publish to"
  value       = module.observability.sns_topic_arn
}
