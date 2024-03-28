module "lambda" {
  source          = "./lambda"
  prefix          = var.prefix
  container_registry = var.container_registry
  container_repository = var.container_repository
  container_image_tag = var.container_image_tag
  ram_mib         = var.lambda_ram_size
  attach_ec2_policy = var.attach_ec2_policy
  attach_cloudwatch_policy = var.attach_cloudwatch_policy
  attach_lambda_policy = var.attach_lambda_policy
  attach_s3_policy = var.attach_s3_policy
  attach_vpc_policy = var.attach_vpc_policy
}
