variable "project" {
  type        = string
  description = "Project name prefix for resource naming"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "region" {
  type        = string
  description = "AWS region (dashboard widgets are regional)"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}

variable "msk_cluster_name" {
  type        = string
  description = "MSK cluster name (CloudWatch `Cluster Name` dimension)"
}

variable "msk_broker_count" {
  type        = number
  default     = 3
  description = "Broker count, used to derive the aggregate CPU threshold"
}

variable "msk_cpu_threshold_pct" {
  type        = number
  default     = 80
  description = "Per-broker CPU% considered hot (summed across brokers for the alarm)"
}

variable "msk_disk_threshold_pct" {
  type        = number
  default     = 85
  description = "KafkaDataLogsDiskUsed % that triggers the disk alarm"
}

variable "redis_cluster_id" {
  type        = string
  description = "ElastiCache cluster ID (CacheClusterId dimension)"
}

variable "redis_memory_threshold_pct" {
  type        = number
  default     = 80
  description = "Redis DatabaseMemoryUsagePercentage alarm threshold"
}

variable "redis_cpu_threshold_pct" {
  type        = number
  default     = 80
  description = "Redis EngineCPUUtilization alarm threshold"
}

variable "alb_name" {
  type        = string
  description = "ALB name (LoadBalancer dimension)"
}

variable "alb_5xx_threshold" {
  type        = number
  default     = 10
  description = "Target 5xx count per 5m window that triggers the ALB alarm"
}

variable "cluster_name" {
  type        = string
  description = "ECS cluster name (ClusterName dimension)"
}

variable "service_names" {
  type        = list(string)
  description = "ECS service names to alarm on (one alarm set per service)"
}

variable "ecs_cpu_threshold_pct" {
  type        = number
  default     = 80
  description = "ECS service CPUUtilization alarm threshold"
}

variable "ecs_memory_threshold_pct" {
  type        = number
  default     = 85
  description = "ECS service MemoryUtilization alarm threshold"
}

variable "alerts_email" {
  type        = string
  default     = ""
  description = "Email for the SNS alert subscription; empty = no email (topic only)"
}
