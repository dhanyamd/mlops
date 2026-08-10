# CloudWatch log groups — one per service so logs are correlated per unit,
# with a retention policy (production logs are cheap; infinite retention is
# not a plan).

resource "aws_cloudwatch_log_group" "flink_jobmanager" {
  name              = "/ecs/${local.name}/flink-jobmanager"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "flink_taskmanager" {
  name              = "/ecs/${local.name}/flink-taskmanager"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "app" {
  for_each          = toset(var.app_service_names)
  name              = "/ecs/${local.name}/${each.value}"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${local.name}/ui"
  retention_in_days = var.log_retention_days
  tags              = local.tags
}
