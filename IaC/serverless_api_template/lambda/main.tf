provider "random" {}

resource "random_id" "random_string" {
  byte_length = 4
}

resource "aws_iam_role" "lambda-role" {
  name = "${var.prefix}-aws-lambda-role-${random_id.random_string.hex}"

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

resource "aws_iam_role_policy_attachment" "eks_workernode_policy" {
  count      = var.attach_eks_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "cloudwatch_policy" {
  count      = var.attach_cloudwatch_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

resource "aws_iam_role_policy_attachment" "ssm_readonly_policy" {
  count      = var.attach_ssm_readonly_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess"
}

resource "aws_iam_role_policy_attachment" "cloudwatchlogs_policy" {
  count      = var.attach_cloudwatch_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "ec2_policy" {
  count      = var.attach_ec2_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
}

resource "aws_iam_role_policy_attachment" "vpc_policy" {
  count      = var.attach_vpc_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonVPCFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_policy" {
  count      = var.attach_s3_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  count      = var.attach_lambda_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSLambda_FullAccess"
}

resource "aws_iam_role_policy_attachment" "iam_policy" {
  count      = var.attach_iam_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/IAMFullAccess"
}

resource "aws_iam_role_policy_attachment" "pricing_policy" {
  count      = var.attach_pricing_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AWSPriceListServiceFullAccess"
}

resource "aws_iam_role_policy_attachment" "admin_policy" {
  count      = var.attach_admin_policy ? 1 : 0
  role       = aws_iam_role.lambda-role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

resource "aws_lambda_function" "lambda" {
  function_name = "${var.prefix}-aws-lambda-${random_id.random_string.hex}"
  package_type  = "Image"
  architectures = ["x86_64"]
  image_uri     = "${var.container_registry}/${var.container_repository}:${var.container_image_tag}"
  memory_size   = var.ram_mib
  timeout       = var.timeout_s
  role          = aws_iam_role.lambda-role.arn

  environment {
    variables = {
      EKS_CLUSTER_NAME                   = var.eks_cluster_name
      DB_API_URL                         = var.db_api_url
      STATE_BUCKET_NAME                  = var.state_bucket_name
      KARPENTER_NODE_ROLE_PARAMETER_NAME = var.karpenter_node_role_parameter_name
      REGION                             = var.region_name
      ECR_URI                            = var.container_registry
      MODEL_S3_URL                       = var.model_s3_url
      UPLOAD_S3_URL                      = var.upload_s3_url
    }
  }
}

resource "aws_cloudwatch_log_group" "lambda-cloudwath-log-group" {
  name              = "/aws/lambda/${aws_lambda_function.lambda.function_name}"
  retention_in_days = 30
}

resource "aws_lambda_function_url" "lambda-url" {
  function_name      = aws_lambda_function.lambda.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = [ "*" ]
    allow_methods = [ "*" ]
    allow_headers = [ "*" ]
  }
}

resource "aws_eks_access_entry" "eks-access-entry" {
  count         = var.attach_eks_policy ? 1 : 0
  cluster_name  = var.eks_cluster_name
  principal_arn = aws_iam_role.lambda-role.arn
}

resource "aws_eks_access_policy_association" "eks-access-policy" {
  count         = var.attach_eks_policy ? 1 : 0
  cluster_name  = var.eks_cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = aws_iam_role.lambda-role.arn

  access_scope {
    type = "cluster"
  }
}
