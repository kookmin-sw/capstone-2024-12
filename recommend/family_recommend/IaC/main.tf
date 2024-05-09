
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

module "recommend_family_1" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_1"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-1"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_1_lambda_function_url" {
  name = "recocommend_family_1_lambda_function_url"
  type = "String"
  value = module.recommend_family_1.function_url

  depends_on = [ module.recommend_family_1 ]
}

module "recommend_family_2" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_2"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-2"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_2_lambda_function_url" {
  name = "recocommend_family_2_lambda_function_url"
  type = "String"
  value = module.recommend_family_2.function_url

  depends_on = [ module.recommend_family_2 ]
}

module "recommend_family_3" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_3"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-3"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_3_lambda_function_url" {
  name = "recocommend_family_3_lambda_function_url"
  type = "String"
  value = module.recommend_family_3.function_url

  depends_on = [ module.recommend_family_3 ]
}

module "recommend_family_4" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_4"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-4"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_4_lambda_function_url" {
  name = "recocommend_family_4_lambda_function_url"
  type = "String"
  value = module.recommend_family_4.function_url

  depends_on = [ module.recommend_family_4 ]
}

module "recommend_family_5" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "recommend_family_5"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "recommend-family-5"
  container_image_tag = "latest"
  lambda_ram_size = 256
  attach_ec2_policy = true
}

resource "aws_ssm_parameter" "param_recommend_family_5_lambda_function_url" {
  name = "recocommend_family_5_lambda_function_url"
  type = "String"
  value = module.recommend_family_5.function_url

  depends_on = [ module.recommend_family_5 ]
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