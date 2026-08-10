output "checkpoint_bucket_name" {
  value = aws_s3_bucket.checkpoints.id
}

output "checkpoint_bucket_arn" {
  value = aws_s3_bucket.checkpoints.arn
}

output "research_bucket_arn" {
  value = aws_s3_bucket.research.arn
}

output "redis_cluster_id" {
  value = aws_elasticache_cluster.redis.id
}

output "redis_endpoint_address" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "redis_port" {
  value = aws_elasticache_cluster.redis.port
}
