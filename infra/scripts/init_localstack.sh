#!/bin/bash
set -x

echo "Initializing LocalStack resources..."

# Configure AWS CLI for LocalStack
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"
alias awslocal="aws --endpoint-url=http://localhost:4566"

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
until awslocal s3 ls; do
  echo "LocalStack not ready yet, retrying in 2 seconds..."
  sleep 2
done

# Create S3 buckets
awslocal s3 mb s3://local-mlflow-artifacts
awslocal s3 mb s3://local-mlops-data

# Create ECR repositories
awslocal ecr create-repository --repository-name local-mlops-fraud-api
awslocal ecr create-repository --repository-name local-mlops-training

echo "LocalStack initialization complete."
