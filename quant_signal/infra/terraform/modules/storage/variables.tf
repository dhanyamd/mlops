variable "project" {
  type        = string
  description = "Project name prefix for resource naming"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnets for the ElastiCache subnet group"
}

variable "internal_security_group_id" {
  type        = string
  description = "SG that allows traffic between platform services"
}

variable "checkpoint_bucket_name" {
  type        = string
  description = "Globally unique S3 bucket for Flink checkpoints"
}

variable "research_bucket_name" {
  type        = string
  description = "Globally unique S3 bucket for research/MLflow artifacts"
}

variable "checkpoint_retention_days" {
  type        = number
  default     = 7
  description = "How long a completed checkpoint is kept in S3"
}

variable "research_ia_days" {
  type        = number
  default     = 30
  description = "Days before research artifacts transition to STANDARD_IA"
}

variable "research_expiration_days" {
  type        = number
  default     = 365
  description = "Days before research artifacts expire"
}

variable "redis_node_type" {
  type        = string
  default     = "cache.t3.micro"
  description = "ElastiCache node type (dev: micro; prod: larger + cluster mode)"
}

variable "redis_engine_version" {
  type        = string
  default     = "7.1"
  description = "ElastiCache Redis engine version"
}

variable "redis_parameter_group" {
  type        = string
  default     = "default.redis7"
  description = "ElastiCache parameter group"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}
