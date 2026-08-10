# ECS module — Fargate cluster + task definitions + services + ALB + Service Connect.
#
# Design notes (research-backed):
# - Fargate is fixed to `networkMode: awsvpc` and ALB target groups use
#   `target_type = "ip"` — there is no host networking to lean on.
# - All tasks run in private subnets; ONLY the ALB is internet-facing. Task SGs
#   never open 0.0.0.0/0 — they rely on the internal SG's self-reference.
# - Service Connect (Envoy sidecar injected by ECS) gives the Flink taskmanager
#   a stable logical name for the jobmanager (`flink-jobmanager`) and lets the
#   UI reach the API by name — no internal ALBs, no hardcoded IPs
#   (tomodahinata Fargate networking guide: prefer Service Connect over Cloud Map
#   when you want retries + metrics; both are in-cluster DNS).
# - Single-task services deploy with minimum=0 / maximum=100: with desired=1 a
#   rolling 100/200 deploy would need a second task that can't start
#   (the classic Fargate "service stuck in DRAINING on a 1-task service" trap).
# - Flink is self-hosted on Fargate (jobmanager + taskmanager, the same two
#   processes docker-compose runs) rather than AWS Managed Flink, so the exact
#   SQL jobs (crypto_features.sql / crypto_features_1h.sql) ship unchanged and
#   checkpointing stays under our control. Checkpoints go to S3
#   (`s3a://…`) — the flink image must bundle the S3 filesystem plugin
#   (flink-s3-fs-hadoop); checkpoint interval 2-5 min is the balanced default
#   (AWS Managed Flink guidance).

locals {
  name = "${var.project}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = var.project
  })

  common_env = [
    { name = "STREAM_KAFKA_BOOTSTRAP_SERVERS", value = var.kafka_bootstrap_plaintext },
    { name = "STREAM_REDIS_URL", value = "redis://${var.redis_endpoint_address}:${var.redis_port}" },
    { name = "STREAM_ENABLED", value = "true" },
    { name = "LOG_LEVEL", value = var.log_level },
  ]

  # The six live agents + the dashboard API — the same processes the local
  # launchd agents run, as Fargate services (one task per agent). Derived from
  # var.app_service_names so the log groups (logs.tf) stay in sync.
  app_service_commands = {
    producer     = ["python", "-m", "stream.producer"]
    materializer = ["python", "-m", "stream.materializer"]
    predictor    = ["python", "-m", "stream.predictor"]
    execution    = ["python", "-m", "stream.execution"]
    simulation   = ["python", "-m", "stream.simulation"]
    watchdog     = ["python", "-m", "scripts.stream_watchdog", "--interval", "60", "--fix"]
    api          = ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  }

  app_services = {
    for name in var.app_service_names : name => {
      command = local.app_service_commands[name]
      port    = name == "api" ? 8000 : null
      connect = name == "api"
    }
  }

  # Bybit demo credentials are the only long-lived secrets the app agents need
  # (stream/execution.py reads BYBIT_DEMO_API_KEY / BYBIT_DEMO_API_SECRET).
  # They are injected via the task definition's `secrets` block (from Secrets
  # Manager, resolved at task start) so they never appear in task env, logs,
  # or the repo. Empty ARNs = no secrets block = the agents run without a
  # broker connection.
  app_secrets = concat(
    var.bybit_demo_api_key_secret_arn != "" ? [
      { name = "BYBIT_DEMO_API_KEY", valueFrom = var.bybit_demo_api_key_secret_arn },
    ] : [],
    var.bybit_demo_api_secret_secret_arn != "" ? [
      { name = "BYBIT_DEMO_API_SECRET", valueFrom = var.bybit_demo_api_secret_secret_arn },
    ] : [],
  )
}

resource "aws_service_discovery_http_namespace" "this" {
  name = local.name
}

resource "aws_ecs_cluster" "this" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  service_connect_defaults {
    namespace = aws_service_discovery_http_namespace.this.arn
  }

  tags = local.tags
}

# ── Flink: jobmanager + taskmanager ──────────────────────────────────────────

resource "aws_ecs_task_definition" "flink_jobmanager" {
  family                   = "${local.name}-flink-jobmanager"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.flink_cpu
  memory                   = var.flink_memory
  execution_role_arn       = var.ecs_execution_role_arn
  task_role_arn            = var.ecs_task_role_arn
  container_definitions = jsonencode([
    {
      name      = "flink-jobmanager"
      image     = var.flink_image
      command   = ["jobmanager"]
      essential = true
      environment = [
        {
          name  = "FLINK_PROPERTIES"
          value = <<-EOT
            jobmanager.rpc.address: flink-jobmanager
            jobmanager.memory.process.size: 1600m
            state.backend: rocksdb
            state.checkpoints.dir: s3a://${var.checkpoint_bucket_name}/flink
            execution.checkpointing.interval: 300000ms
          EOT
        },
      ]
      portMappings = [
        { containerPort = 8081, protocol = "tcp", name = "flink-rpc" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.flink_jobmanager.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "flink"
        }
      }
    },
  ])
  tags = local.tags
}

resource "aws_ecs_task_definition" "flink_taskmanager" {
  family                   = "${local.name}-flink-taskmanager"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.flink_cpu
  memory                   = var.flink_memory
  execution_role_arn       = var.ecs_execution_role_arn
  task_role_arn            = var.ecs_task_role_arn
  container_definitions = jsonencode([
    {
      name      = "flink-taskmanager"
      image     = var.flink_image
      command   = ["taskmanager"]
      essential = true
      environment = [
        {
          name  = "FLINK_PROPERTIES"
          value = <<-EOT
            jobmanager.rpc.address: flink-jobmanager
            taskmanager.numberOfTaskSlots: 4
            taskmanager.memory.process.size: 1600m
            state.backend: rocksdb
            state.checkpoints.dir: s3a://${var.checkpoint_bucket_name}/flink
          EOT
        },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.flink_taskmanager.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "flink"
        }
      }
    },
  ])
  tags = local.tags
}

resource "aws_ecs_service" "flink_jobmanager" {
  name                   = "flink-jobmanager"
  cluster                = aws_ecs_cluster.this.id
  task_definition        = aws_ecs_task_definition.flink_jobmanager.arn
  desired_count          = 1
  launch_type            = "FARGATE"
  enable_execute_command = true

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.internal_security_group_id]
    assign_public_ip = false
  }

  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_http_namespace.this.arn

    service {
      port_name      = "flink-rpc"
      discovery_name = "flink-jobmanager"
      client_alias {
        dns_name = "flink-jobmanager"
        port     = 8081
      }
    }
  }

  tags = local.tags
}

resource "aws_ecs_service" "flink_taskmanager" {
  name            = "flink-taskmanager"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.flink_taskmanager.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.internal_security_group_id]
    assign_public_ip = false
  }

  service_connect_configuration {
    # Client-only participant: the taskmanager resolves `flink-jobmanager` via
    # the namespace but exposes no endpoint of its own, so it registers NO
    # port/alias. (Registering a client_alias here with no matching portMapping
    # is the classic invalid ServiceConnect config that fails the service.)
    enabled   = true
    namespace = aws_service_discovery_http_namespace.this.arn
  }

  tags = local.tags
}

# ── App agents + API (Fargate services from the app image) ───────────────────

resource "aws_ecs_task_definition" "app" {
  for_each                 = local.app_services
  family                   = "${local.name}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = var.ecs_execution_role_arn
  task_role_arn            = var.ecs_task_role_arn
  container_definitions = jsonencode([
    {
      name        = each.key
      image       = var.app_image
      command     = each.value.command
      essential   = true
      environment = local.common_env
      secrets     = local.app_secrets
      portMappings = lookup(each.value, "port", null) != null ? [
        { containerPort = each.value.port, protocol = "tcp", name = "${each.key}-http" },
      ] : []
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app[each.key].name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = each.key
        }
      }
    },
  ])
  tags = local.tags
}

resource "aws_ecs_service" "app" {
  for_each        = local.app_services
  name            = each.key
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app[each.key].arn
  desired_count   = 1
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.internal_security_group_id]
    assign_public_ip = false
  }

  dynamic "service_connect_configuration" {
    for_each = lookup(each.value, "connect", false) ? [1] : []
    content {
      enabled   = true
      namespace = aws_service_discovery_http_namespace.this.arn

      service {
        port_name      = "${each.key}-http"
        discovery_name = each.key
        client_alias {
          dns_name = each.key
          port     = each.value.port
        }
      }
    }
  }

  tags = local.tags
}

# ── UI (Next.js) behind the ALB ──────────────────────────────────────────────

resource "aws_ecs_task_definition" "ui" {
  family                   = "${local.name}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ui_cpu
  memory                   = var.ui_memory
  execution_role_arn       = var.ecs_execution_role_arn
  task_role_arn            = var.ecs_task_role_arn
  container_definitions = jsonencode([
    {
      name      = "ui"
      image     = var.ui_image
      command   = ["npm", "run", "start"]
      essential = true
      environment = [
        # next.config.ts rewrites /api/* to API_BASE_URL; the Service Connect
        # name `api` resolves in-cluster, so no public API exposure is needed.
        { name = "API_BASE_URL", value = "http://api:8000" },
        { name = "PORT", value = "3000" },
      ]
      portMappings = [
        { containerPort = 3000, protocol = "tcp", name = "ui-http" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ui.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "ui"
        }
      }
    },
  ])
  tags = local.tags
}

resource "aws_ecs_service" "ui" {
  name            = "ui"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.internal_security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 3000
  }

  tags = local.tags
}
