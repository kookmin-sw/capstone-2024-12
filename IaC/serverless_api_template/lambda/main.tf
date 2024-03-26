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
