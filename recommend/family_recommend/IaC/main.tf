
# module "cpu_family_recommend" {
#   source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
#   prefix = "cpu_family_recommend"
#   container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
#   container_repository = "recommend-inference-cpu-family"
#   container_image_tag = "latest"
#   lambda_ram_size = 256
#   attach_s3_policy = true
#   attach_ec2_policy = true
# }

# resource "aws_ssm_parameter" "param_cpu_recommend_lambda_function_url" {
#   name = "cpu_recommend_lambda_function_url"
#   type = "String"
#   value = module.cpu_family_recommend.function_url

#   depends_on = [ module.cpu_family_recommend ]
# }

module "recommend_family" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family"
  container_registry = var.container_registry
  container_repository = "recommend-family"
  container_image_tag = "latest"
  lambda_ram_size = 256
  lambda_timeout = 240
  attach_ec2_policy = true
  attach_pricing_policy = true
  attach_admin_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_lambda_function_url" {
  name = "recommend_family_lambda_function_url"
  type = "String"
  value = module.recommend_family.function_url

  depends_on = [ module.recommend_family ]
}

variable "nodepool_numbers" {
  type    = list(string)
  default = ["1", "2", "3", "4", "5"]
}

resource "aws_ssm_parameter" "nodepool_ondemand_price" {
  for_each = toset(var.nodepool_numbers)
  
  name  = "nodepool_${each.value}_ondemand_price"
  type  = "String"
  value = "9999"
}

resource "aws_ssm_parameter" "nodepool_spot_price" {
  for_each = toset(var.nodepool_numbers)
  
  name  = "nodepool_${each.value}_spot_price"
  type  = "String"
  value = "9999"
}

# provider "aws" {
#     region = var.region
#     profile = var.awscli_profile
# }

terraform {
  backend "s3" {
    bucket = "sskai-terraform-state"
    key = "family_recommend/tf.state"
    region = "ap-northeast-2"
    encrypt = true
  }
}