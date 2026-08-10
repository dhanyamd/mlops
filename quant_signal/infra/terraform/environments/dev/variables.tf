variable "project" {
  type        = string
  default     = "quant-signal"
  description = "Prefix for all resource names"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment (dev/stage/prod)"
}

variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for the whole stack"
}

# ── Networking ───────────────────────────────────────────────────────────────

variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "VPC CIDR block"
}

variable "azs" {
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
  description = "AZs to spread subnets + MSK brokers across"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
  description = "One public subnet CIDR per AZ"
}

variable "private_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
  description = "One private subnet CIDR per AZ"
}

# ── Images (CI builds + pushes these; the tag pins what runs) ───────────────

variable "app_image_tag" {
  type        = string
  default     = "latest"
  description = "Tag of the app image to run; CI sets this to the commit SHA"
}

variable "flink_image_tag" {
  type        = string
  default     = "latest"
  description = "Tag of the flink image to run"
}

variable "ui_image_tag" {
  type        = string
  default     = "latest"
  description = "Tag of the UI image to run"
}

# ── Runtime wiring ───────────────────────────────────────────────────────────

variable "app_service_names" {
  type        = list(string)
  default     = ["producer", "materializer", "predictor", "execution", "simulation", "watchdog", "api"]
  description = "Agents + API to run as Fargate services (must exist in the ecs module command map)"
}

variable "log_level" {
  type        = string
  default     = "INFO"
  description = "LOG_LEVEL for the app agents"
}

variable "bybit_demo_api_key_secret_arn" {
  type        = string
  default     = ""
  description = "Secrets Manager ARN for BYBIT_DEMO_API_KEY (leave empty to run without broker creds)"
}

variable "bybit_demo_api_secret_secret_arn" {
  type        = string
  default     = ""
  description = "Secrets Manager ARN for BYBIT_DEMO_API_SECRET"
}

variable "alerts_email" {
  type        = string
  default     = ""
  description = "Email for CloudWatch alarm notifications (empty = topic only)"
}
