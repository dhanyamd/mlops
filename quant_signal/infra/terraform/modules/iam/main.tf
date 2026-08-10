# IAM module — the two-role ECS split.
#
# Research-backed design (ECS on Fargate reference architecture):
# - Execution role: assumed by the Fargate agent — pull from ECR, ship logs to
#   CloudWatch, read the secrets injected at startup. It should NOT hold the
#   app's business-logic permissions.
# - Task role: assumed by the app inside the container — least privilege for
#   the data path (S3 checkpoints/artifacts, Secrets Manager). Each service
#   *could* get its own task role; a shared one with S3 + secrets is the
#   pragmatic dev profile, with the blast-radius split documented as the
#   production upgrade (per-service roles, per-service S3 prefixes).
# - SSM: ManagedInstanceCore lets Fargate tasks run `ecs exec` — how the
#   runbook submits the Flink SQL jobs to the jobmanager from the task itself.

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })
}

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Secrets access is a separate policy so it can be skipped entirely when no
# secrets are provisioned (the dev profile runs agents without Bybit creds).
# An empty `Resource` list would be rejected by AWS, so the policy and its
# attachment are count-guarded on var.secret_arns.
resource "aws_iam_policy" "execution_secrets" {
  count       = length(var.secret_arns) > 0 ? 1 : 0
  name        = "${local.name}-ecs-execution-secrets"
  description = "Read the env secrets injected into task definitions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "ssm:GetParameter",
        ]
        Resource = var.secret_arns
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "execution_secrets" {
  count      = length(var.secret_arns) > 0 ? 1 : 0
  role       = aws_iam_role.ecs_execution.name
  policy_arn = aws_iam_policy.execution_secrets[0].arn
}

resource "aws_iam_role" "ecs_task" {
  name               = "${local.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

resource "aws_iam_policy" "task_data" {
  name        = "${local.name}-ecs-task-data"
  description = "Least-privilege data path: S3 checkpoints/artifacts + secrets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Effect = "Allow"
          Action = [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject",
          ]
          Resource = [
            for arn in var.bucket_arns : "${arn}/*"
          ]
        },
        {
          Effect   = "Allow"
          Action   = ["s3:ListBucket"]
          Resource = var.bucket_arns
        },
      ],
      # Only grant secret read when the caller actually provisioned secrets.
      length(var.secret_arns) > 0 ? [
        {
          Effect   = "Allow"
          Action   = ["secretsmanager:GetSecretValue"]
          Resource = var.secret_arns
        },
      ] : [],
    )
  })
}

resource "aws_iam_role_policy_attachment" "task_data" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.task_data.arn
}

resource "aws_iam_role_policy_attachment" "task_ssm" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
