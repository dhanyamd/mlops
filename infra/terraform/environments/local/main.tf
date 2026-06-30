terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  
  endpoints {
    s3  = "http://localhost:4566"
    ecr = "http://localhost:4566"
    ecs = "http://localhost:4566"
    ec2 = "http://localhost:4566"
  }
}

module "networking" {
  source      = "../../modules/networking"
  environment = "local"
  vpc_cidr    = "10.0.0.0/16"
}

module "ecs" {
  source      = "../../modules/ecs"
  environment = "local"
}

module "s3_mlflow" {
  source      = "../../modules/s3"
  environment = "local"
  bucket_name = "mlflow-artifacts"
}

module "s3_data" {
  source      = "../../modules/s3"
  environment = "local"
  bucket_name = "mlops-data"
}

module "ecr_api" {
  source          = "../../modules/ecr"
  environment     = "local"
  repository_name = "mlops-fraud-api"
}

module "ecr_training" {
  source          = "../../modules/ecr"
  environment     = "local"
  repository_name = "mlops-training"
}
