variable "environment" {
  description = "Environment name (e.g., local, dev, prod)"
  type        = string
}

variable "repository_name" {
  description = "Name of the ECR repository"
  type        = string
}

resource "aws_ecr_repository" "repo" {
  name                 = "${var.environment}-${var.repository_name}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
  }
}

output "repository_url" {
  value = aws_ecr_repository.repo.repository_url
}
