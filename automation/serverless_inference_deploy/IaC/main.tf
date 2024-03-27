
module "serverless_inference_deploy" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "serverless_inference_deploy"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "serverless-inference-deploy"
  container_image_tag = "latest"
  lambda_ram_size = 2048
  attach_s3_policy = true
  attach_ec2_policy = true
  attach_lambda_policy = true
  attach_cloudwatch_policy = true
}

variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "awscli_profile" {
  type    = string
  default = "default"
}

output "function_url" {
  value = module.serverless_inference.function_url
}

provider "aws" {
    region = var.region
    profile = var.awscli_profile
}

terraform {
  backend "s3" {
    bucket = "sskai-terraform-state"
    key = "serverless_inference_deploy/tf.state"
    region = "ap-northeast-2"
    encrypt = true
  }
}