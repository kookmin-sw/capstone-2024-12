resource "null_resource" "download_lambda_codes" {
    count = length(var.api_list)
  provisioner "local-exec" {
    command = <<-EOT
mkdir /tmp/${var.api_list[count.index]}
curl -o /tmp/${var.api_list[count.index]}/index.mjs https://raw.githubusercontent.com/kookmin-sw/capstone-2024-12/master/backend/${var.api_list[count.index]}/index.mjs
zip -j /tmp/${var.api_list[count.index]}.zip /tmp/${var.api_list[count.index]}/index.mjs
EOT
  }
}

resource "aws_iam_role" "lambda_api_role" {
  name = "${var.region}-sskai-db-api-lambda-role"

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
  role       = aws_iam_role.lambda_api_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "cloudwatch_policy" {
  role       = aws_iam_role.lambda_api_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

resource "aws_iam_role_policy_attachment" "cloudwatchlogs_policy" {
  role       = aws_iam_role.lambda_api_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_policy" {
  role       = aws_iam_role.lambda_api_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "dynamodb_policy" {
  role       = aws_iam_role.lambda_api_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess"
}