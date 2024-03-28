resource "aws_iam_role" "lambda-role" {
  name = "${var.prefix}-aws-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Sid    = ""
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_policy" {
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "cloudwatch_policy" {
  count      = var.attach_cloudwatch_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/CloudWatchFullAccess"
}

resource "aws_iam_role_policy_attachment" "cloudwatchlogs_policy" {
  count      = var.attach_cloudwatch_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_policy" {
  count      = var.attach_ec2_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2FullAccess"
}

resource "aws_iam_role_policy_attachment" "vpc_policy" {
  count      = var.attach_vpc_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonVPCFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_policy" {
  count      = var.attach_s3_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  count      = var.attach_lambda_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambda_FullAccess"
}

resource "aws_lambda_function" "lambda" {
  function_name = "${var.prefix}-aws-lambda"
  package_type  = "Image"
  architectures = ["x86_64"]
  image_uri     = "${var.container_registry}/${var.container_repository}:${var.container_image_tag}"
  memory_size   = var.ram_mib
  timeout       = 120
  role          = aws_iam_role.lambda-role.arn

}

resource "aws_cloudwatch_log_group" "lambda-cloudwath-log-group" {
  name              = "/aws/lambda/${aws_lambda_function.lambda.function_name}"
  retention_in_days = 30
}

resource "aws_lambda_function_url" "lambda-url" {
  function_name      = aws_lambda_function.lambda.function_name
  authorization_type = "NONE"
}