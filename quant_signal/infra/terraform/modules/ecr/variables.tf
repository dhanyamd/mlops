variable "project" {
  type        = string
  description = "Project name prefix for resource naming"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
}

variable "repositories" {
  type        = list(string)
  description = "Image repos to create (e.g. app, flink, ui)"
  default     = ["app", "flink", "ui"]
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}
