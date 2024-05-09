
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

module "recommend_family_1" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_1"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-1"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

# resource "aws_ssm_parameter" "param_cpu_recommend_lambda_function_url" {
#   name = "cpu_recommend_lambda_function_url"
#   type = "String"
#   value = module.cpu_family_recommend.function_url

#   depends_on = [ module.cpu_family_recommend ]
# }

resource "aws_ssm_parameter" "param_recommend_family_1_lambda_function_url" {
  name = "recocommend_family_1_lambda_function_url"
  type = "String"
  value = module.recommend_family_1.function_url

  depends_on = [ module.recommend_family_1 ]
}

provider "aws" {
    region = var.region
    profile = var.awscli_profile
}

terraform {
  backend "s3" {
    bucket = "sskai-terraform-state"
    key = "family_recommend/tf.state"
    region = "ap-northeast-2"
    encrypt = true
  }
}