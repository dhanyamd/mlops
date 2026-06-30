variable "environment" {
  description = "Environment name (e.g., local, dev, prod)"
  type        = string
}

resource "aws_ecs_cluster" "mlops_cluster" {
  name = "${var.environment}-mlops-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = var.environment
  }
}

output "cluster_id" {
  value = aws_ecs_cluster.mlops_cluster.id
}

output "cluster_name" {
  value = aws_ecs_cluster.mlops_cluster.name
}
