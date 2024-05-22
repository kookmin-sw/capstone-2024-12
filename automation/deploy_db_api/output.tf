output "api_endpoint_url" {
  value = aws_apigatewayv2_stage.apigtw_stage.invoke_url
}

output "model_storage_bucket_name" {
  value = aws_s3_bucket.sskai-s3-model-bucket.bucket
}