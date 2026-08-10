# quant-signal AWS infrastructure (Terraform)

Managed Kafka (MSK) → Flink on Fargate → ElastiCache Redis (online store) →
ECS app agents + Next.js UI behind an ALB — all as reusable modules.

```
terraform/
├── modules/
│   ├── networking/     VPC, subnets, NAT, route tables, SGs (private+tasks)
│   ├── storage/        S3 (flink checkpoints, research artifacts) + ElastiCache Redis
│   ├── msk/            MSK cluster (3 brokers, TLS_PLAINTEXT, per-broker monitoring)
│   ├── iam/            ECS execution role + least-privilege task role
│   ├── ecr/            One registry repo per image (app / flink / ui)
│   ├── ecs/            Cluster, Service Connect namespace, task defs + services, ALB
│   └── observability/  CloudWatch alarms (MSK/Redis/ALB/ECS) + SNS + dashboard
└── environments/
    └── dev/            Real deployment root: S3 backend + lock, module wiring
```

## Design notes (research-backed)

- **Fargate + Service Connect** for inter-service DNS (`flink-jobmanager`,
  `api`). The taskmanager joins the namespace as a *client only* — it resolves
  `flink-jobmanager` but registers no endpoint of its own (AWS ECS docs:
  `services` is not required for a client service in a namespace).
- **Flink is self-hosted on Fargate** (jobmanager + taskmanager) so the exact
  SQL jobs ship unchanged; checkpoints go to S3 (`s3a://`), which requires the
  `flink-s3-fs-hadoop` plugin **under `plugins/`, not `lib/`** (Flink 1.19 docs).
  Authorization comes from the task role, never baked-in keys.
- **Two ECS roles**: execution role (ECR pulls, logs, secrets at startup) vs
  task role (S3 + Secrets Manager for the app). The task role does not hold
  registry/log permissions.
- **3 MSK brokers** minimum — `replication.factor 3 / min.insync.replicas 2`
  is the durability floor; one broker is not a real Kafka cluster.
- **Single-NAT dev profile** (production-shaped, cheaper); per-AZ NAT is the
  prod upgrade.

## Prerequisites

- Terraform ≥ 1.9, AWS CLI, credentials for an account where you own IAM.
- `quant_signal/` CI has built + pushed the three images to ECR (or run
  `docker build`/`docker push` manually first).

## Bootstrap (one-time)

State backend (S3 bucket + DynamoDB lock table) is created out-of-band so the
first `apply` has somewhere to write state:

```bash
aws s3api create-bucket --bucket quant-signal-tfstate --region us-east-1 \
  --create-bucket-configuration LocationConstraint=us-east-1
aws s3api put-bucket-versioning --bucket quant-signal-tfstate \
  --versioning-configuration Status=Enabled
aws dynamodb create-table --table-name quant-signal-tfstate-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1
```

## Deploy

```bash
cd environments/dev
terraform init                 # configures the S3 backend + downloads providers
cp terraform.tfvars.example terraform.tfvars   # fill in as needed
terraform plan
terraform apply
```

Outputs give you the front door and connection details:

```bash
terraform output alb_dns_name
terraform output msk_bootstrap_brokers
terraform output redis_endpoint
```

## Secrets (Bybit demo credentials)

The app agents read `BYBIT_DEMO_API_KEY` / `BYBIT_DEMO_API_SECRET` from
Secrets Manager (injected as task-definition `secrets`, never in env or logs):

```bash
aws secretsmanager create-secret \
  --name quant-signal/bybit-demo-api-key \
  --secret-string "your-demo-api-key"
aws secretsmanager create-secret \
  --name quant-signal/bybit-demo-api-secret \
  --secret-string "your-demo-api-secret"
```

Put the two ARNs in `terraform.tfvars`
(`bybit_demo_api_key_secret_arn`, `bybit_demo_api_secret_secret_arn`) and
`terraform apply`. The IAM policies and task definitions pick them up. Leave
them empty to run the stack without broker credentials.

## Deploying the Flink SQL jobs

The jobmanager is reachable via Service Connect at `flink-jobmanager:8081`.
From your laptop (SSM port-forward through the cluster, or `ecs exec` on the
jobmanager task):

```bash
CLUSTER=$(terraform output -raw ecs_cluster_name)
aws ecs update-service --cluster "$CLUSTER" --service flink-jobmanager \
  --enable-execute-command --force-new-deployment

aws ecs execute-command --cluster "$CLUSTER" --task <task-id> \
  --container flink-jobmanager --interactive --command "/bin/bash"

# inside the container, submit each SQL job:
./bin/flink run -d /opt/flink/jobs/crypto_features.sql
./bin/flink run -d /opt/flink/jobs/crypto_features_1h.sql
```

Checkpoints are restored from `s3a://quant-signal-dev-flink-checkpoints/flink`
on restart, so a resubmit resumes rather than replays.

## Operations

- **Logs:** each service writes to its own CloudWatch group under
  `/ecs/<env>-<project>/…`.
- **Alarms:** `quant-signal-dev-*` CloudWatch alarms + the
  `quant-signal-dev-platform` dashboard. All alarm → SNS
  (`quant-signal-dev-alerts`); subscribe email/chat in the AWS console.
- **Updates:** `terraform apply` after any change; tasks are recreated by the
  task-definition revision bump.

## Teardown

```bash
terraform destroy          # everything Terraform manages
# delete state + lock afterwards if you truly want to be rid of it:
aws s3 rb s3://quant-signal-tfstate --force
aws dynamodb delete-table --table-name quant-signal-tfstate-lock
```
