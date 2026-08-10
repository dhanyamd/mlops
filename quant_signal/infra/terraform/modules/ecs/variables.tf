variable "project" {
  type        = string
  description = "Project name prefix for resource naming"
}

variable "environment" {
  type        = string
  description = "Deployment environment (e.g. local, dev, prod)"
}

variable "region" {
  type        = string
  description = "AWS region (log groups + ECR pulls are regional)"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}

# ── Network wiring (from the networking module) ──────────────────────────────

variable "vpc_id" {
  type        = string
  description = "VPC the ALB + tasks live in"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnets for the internet-facing ALB"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnets for Fargate tasks"
}

variable "internal_security_group_id" {
  type        = string
  description = "SG shared by all platform services (self-referencing)"
}

variable "alb_security_group_id" {
  type        = string
  description = "Internet-facing ALB SG"
}

# ── IAM (from the iam module) ────────────────────────────────────────────────

variable "ecs_execution_role_arn" {
  type        = string
  description = "Execution role ARN (Fargate agent: ECR pulls, logs, secrets)"
}

variable "ecs_task_role_arn" {
  type        = string
  description = "Task role ARN (app inside container: S3 checkpoints/artifacts)"
}

# ── Images (built + pushed by CI to ECR) ─────────────────────────────────────

variable "app_image" {
  type        = string
  description = "App image URL (all Python agents + API are commands of this image)"
}

variable "flink_image" {
  type        = string
  description = "Flink image URL; must bundle the Kafka SQL connector + flink-s3-fs-hadoop"
}

variable "ui_image" {
  type        = string
  description = "Next.js UI image URL (served behind the ALB)"
}

# ── App runtime wiring (matches the env contract in config/settings.py) ─────

variable "kafka_bootstrap_plaintext" {
  type        = string
  description = "MSK PLAINTEXT bootstrap brokers, e.g. b-1.….kafka.us-east-1.amazonaws.com:9098"
}

variable "redis_endpoint_address" {
  type        = string
  description = "ElastiCache Redis hostname (online store)"
}

variable "redis_port" {
  type        = number
  description = "ElastiCache Redis port"
}

variable "log_level" {
  type        = string
  default     = "INFO"
  description = "STREAM_LOG_LEVEL for the app agents"
}

variable "app_service_names" {
  type        = list(string)
  description = "The agents + API to run as Fargate services (one task per service)"
}

variable "bybit_demo_api_key_secret_arn" {
  type        = string
  default     = ""
  description = "Secrets Manager ARN for BYBIT_DEMO_API_KEY; empty = no Bybit credentials injected"
}

variable "bybit_demo_api_secret_secret_arn" {
  type        = string
  default     = ""
  description = "Secrets Manager ARN for BYBIT_DEMO_API_SECRET; empty = no Bybit credentials injected"
}

# ── Task sizing (Fargate CPU/MEMORY pairs are a fixed menu) ──────────────────

variable "flink_cpu" {
  type        = number
  default     = 1024
  description = "Fargate CPU units for jobmanager + taskmanager"
}

variable "flink_memory" {
  type        = number
  default     = 2048
  description = "Fargate memory (MiB) for jobmanager + taskmanager"
}

variable "app_cpu" {
  type        = number
  default     = 256
  description = "Fargate CPU units for app agents"
}

variable "app_memory" {
  type        = number
  default     = 512
  description = "Fargate memory (MiB) for app agents"
}

variable "ui_cpu" {
  type        = number
  default     = 256
  description = "Fargate CPU units for the UI"
}

variable "ui_memory" {
  type        = number
  default     = 512
  description = "Fargate memory (MiB) for the UI"
}

# ── Storage + logging ────────────────────────────────────────────────────────

variable "checkpoint_bucket_name" {
  type        = string
  description = "S3 bucket for Flink checkpoints (s3a:// URI)"
}

variable "log_retention_days" {
  type        = number
  default     = 30
  description = "CloudWatch log group retention"
}
