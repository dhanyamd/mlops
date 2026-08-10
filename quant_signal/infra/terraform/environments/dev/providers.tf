# Provider config — defaults for the whole environment. default_tags here are
# the base every resource inherits; modules layer Name/Environment on top.
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "terraform"
      Project     = var.project
    }
  }
}
