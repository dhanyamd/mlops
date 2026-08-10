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
  description = "Private subnets the brokers run in"
}

variable "internal_security_group_id" {
  type        = string
  description = "SG attached to brokers; platform tasks share it"
}

variable "kafka_version" {
  type        = string
  default     = "3.7.0"
  description = "Apache Kafka version on MSK"
}

variable "broker_instance_type" {
  type        = string
  default     = "kafka.m5.large"
  description = "Broker instance type"
}

variable "broker_count" {
  type        = number
  default     = 3
  description = "Number of brokers (3 = smallest honest cluster)"
}

variable "broker_storage_gb" {
  type        = number
  default     = 100
  description = "EBS volume size per broker (GB)"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Extra tags merged onto every resource"
}
