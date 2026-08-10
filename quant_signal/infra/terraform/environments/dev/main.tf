# Dev environment — the whole platform as modules wired in dependency order:
# networking → storage/msk (need subnets+SG) → iam (needs bucket ARNs) →
# ecr → ecs (needs everything) → observability (watches what ecs/msk/storage
# created). `depends_on` is avoided on purpose — the data flow is explicit.

locals {
  bucket_arns = [
    module.storage.checkpoint_bucket_arn,
    module.storage.research_bucket_arn,
  ]

  # Only provisioned secrets reach the IAM policy + task definitions.
  secret_arns = compact([
    var.bybit_demo_api_key_secret_arn,
    var.bybit_demo_api_secret_secret_arn,
  ])
}

module "networking" {
  source      = "../../modules/networking"
  project     = var.project
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
  azs         = var.azs

  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
}

module "storage" {
  source      = "../../modules/storage"
  project     = var.project
  environment = var.environment

  checkpoint_bucket_name = "${var.project}-${var.environment}-flink-checkpoints"
  research_bucket_name   = "${var.project}-${var.environment}-research"

  private_subnet_ids         = module.networking.private_subnet_ids
  internal_security_group_id = module.networking.internal_security_group_id
}

module "msk" {
  source      = "../../modules/msk"
  project     = var.project
  environment = var.environment

  private_subnet_ids         = module.networking.private_subnet_ids
  internal_security_group_id = module.networking.internal_security_group_id
}

module "iam" {
  source      = "../../modules/iam"
  project     = var.project
  environment = var.environment

  bucket_arns = local.bucket_arns
  secret_arns = local.secret_arns
}

module "ecr" {
  source      = "../../modules/ecr"
  project     = var.project
  environment = var.environment

  repositories = ["app", "flink", "ui"]
}

module "ecs" {
  source      = "../../modules/ecs"
  project     = var.project
  environment = var.environment
  region      = var.region

  vpc_id                     = module.networking.vpc_id
  public_subnet_ids          = module.networking.public_subnet_ids
  private_subnet_ids         = module.networking.private_subnet_ids
  internal_security_group_id = module.networking.internal_security_group_id
  alb_security_group_id      = module.networking.alb_security_group_id

  app_image   = "${module.ecr.repository_urls["app"]}:${var.app_image_tag}"
  flink_image = "${module.ecr.repository_urls["flink"]}:${var.flink_image_tag}"
  ui_image    = "${module.ecr.repository_urls["ui"]}:${var.ui_image_tag}"

  kafka_bootstrap_plaintext = module.msk.bootstrap_brokers_plaintext
  redis_endpoint_address    = module.storage.redis_endpoint_address
  redis_port                = module.storage.redis_port
  log_level                 = var.log_level
  app_service_names         = var.app_service_names

  bybit_demo_api_key_secret_arn    = var.bybit_demo_api_key_secret_arn
  bybit_demo_api_secret_secret_arn = var.bybit_demo_api_secret_secret_arn

  ecs_execution_role_arn = module.iam.ecs_execution_role_arn
  ecs_task_role_arn      = module.iam.ecs_task_role_arn

  checkpoint_bucket_name = module.storage.checkpoint_bucket_name
}

module "observability" {
  source      = "../../modules/observability"
  project     = var.project
  environment = var.environment
  region      = var.region

  msk_cluster_name = module.msk.cluster_name
  redis_cluster_id = module.storage.redis_cluster_id
  alb_name         = module.ecs.alb_name
  cluster_name     = module.ecs.cluster_name
  service_names    = module.ecs.service_names

  alerts_email = var.alerts_email
}
