variable "project" {
  type        = string
  description = "Project name prefix for resource naming"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "bucket_arns" {
  type        = list(string)
  description = "S3 bucket ARNs the task role can read/write"
  default     = []
}

variable "secret_arns" {
  type        = list(string)
  description = "Secrets Manager / SSM ARNs the roles may read"
  default     = []
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}
