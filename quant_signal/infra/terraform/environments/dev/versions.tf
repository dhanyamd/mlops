terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # State lives in S3 with DynamoDB locking — never local state for an
  # environment that more than one person can touch. The backend bucket is
  # created out-of-band (see infra/terraform/README.md) so the first `apply`
  # has somewhere to write state.
  backend "s3" {
    bucket         = "quant-signal-tfstate"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "quant-signal-tfstate-lock"
    encrypt        = true
  }
}
