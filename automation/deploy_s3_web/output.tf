output "s3_website_url" {
 value = aws_s3_bucket_website_configuration.sskai-s3-web-bucket-web-conf.website_endpoint 
}

output "website_bucket_name" {
  value = aws_s3_bucket.sskai-s3-web-bucket.bucket
}
