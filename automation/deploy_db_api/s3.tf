
resource "aws_s3_bucket" "sskai-s3-model-bucket" {
  bucket        = "sskai-s3-model-${random_id.random_string.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "sskai-s3-model-bucket-public-conf" {
  bucket = aws_s3_bucket.sskai-s3-model-bucket.bucket

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "sskai-s3-model-bucket-policy" {
  bucket = aws_s3_bucket.sskai-s3-model-bucket.bucket
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = "*",
        Action = "s3:GetObject",
        Resource = "${aws_s3_bucket.sskai-s3-model-bucket.arn}/*"
      }
    ]
  })
}

resource "aws_s3_bucket_cors_configuration" "sskai-s3-model-bucket-cors-conf" {
  bucket = aws_s3_bucket.sskai-s3-model-bucket.bucket

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
  }
  
}