output "repository_urls" {
  description = "Map of repo name → ECR registry URL"
  value       = { for k, r in aws_ecr_repository.this : k => r.repository_url }
}

output "repository_arns" {
  description = "Map of repo name → ARN"
  value       = { for k, r in aws_ecr_repository.this : k => r.arn }
}
