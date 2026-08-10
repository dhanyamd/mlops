output "cluster_name" {
  description = "ECS cluster name (for `aws ecs` CLI + runbook)"
  value       = aws_ecs_cluster.this.name
}

output "cluster_id" {
  description = "ECS cluster ID"
  value       = aws_ecs_cluster.this.id
}

output "alb_dns_name" {
  description = "Public DNS of the internet-facing ALB (UI front door)"
  value       = aws_lb.this.dns_name
}

output "alb_name" {
  description = "ALB name (CloudWatch LoadBalancer dimension)"
  value       = aws_lb.this.name
}

output "service_names" {
  description = "All Fargate service names registered on the cluster"
  value = concat(
    [for k in keys(local.app_services) : k],
    ["flink-jobmanager", "flink-taskmanager", "ui"],
  )
}
