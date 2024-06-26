resource "aws_lambda_function" "sskai-ddb-data-api" {
    function_name = "sskai-ddb-data-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-data-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-ddb-inferences-api" {
    function_name = "sskai-ddb-inferences-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-inferences-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-ddb-logs-api" {
    function_name = "sskai-ddb-logs-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-logs-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-ddb-models-api" {
    function_name = "sskai-ddb-models-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-models-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-ddb-trains-api" {
    function_name = "sskai-ddb-trains-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-trains-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-ddb-users-api" {
    function_name = "sskai-ddb-users-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-ddb-users-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-s3-multipart-presigned-url" {
    function_name = "sskai-s3-multipart-presigned-url-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-s3-multipart-presigned-url.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-s3-presigned-url-api" {
    function_name = "sskai-s3-presigned-url-api-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-s3-presigned-url-api.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}

resource "aws_lambda_function" "sskai-cost-calculate-api" {
    function_name = "sskai-cost-calculate-${random_id.random_string.hex}"
    filename      = "/tmp/sskai-cost-calculate.zip"
    role          = aws_iam_role.lambda_api_role.arn
    handler       = "index.handler"
    runtime       = "nodejs20.x"
    memory_size   = 128
    timeout       = 60
    environment {
      variables = {
        BUCKET_NAME = aws_s3_bucket.sskai-s3-model-bucket.bucket
      }
    }
    depends_on = [ null_resource.download_lambda_codes ]
}