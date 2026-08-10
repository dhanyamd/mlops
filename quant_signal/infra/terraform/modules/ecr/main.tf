# ECR module — per-image repositories with lifecycle policies.
#
# One repo per *image* (app, flink, ui), not per service: the app agents are
# different commands of the same image, so separate repos would mean building
# the same Python env N times. scan_on_push + a lifecycle policy (keep 10
# tagged, expire untagged) keeps the registry from becoming a landfill
# (m-saad-siddique ECS blueprint: lifecycle policies + scan on push are table
# stakes for a production registry).

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })
}

resource "aws_ecr_repository" "this" {
  for_each             = toset(var.repositories)
  name                 = "${local.name}-${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(local.tags, { Name = "${local.name}-${each.key}" })
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "expire untagged images"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "keep last 10 tagged images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}
