# prefix, container_repository 변경 (대채적인 이름)

module "llama_inference_deploy" {
  source                     = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix                     = "llama-inference-deploy"
  container_registry         = var.container_registry
  container_repository       = "llama-inference-deploy"
  container_image_tag        = "latest"
  lambda_ram_size            = 2048
  attach_s3_policy           = true
  attach_ec2_policy          = true
  attach_eks_policy          = true
  attach_ssm_readonly_policy = true
  region_name = var.region
  eks_cluster_name = var.eks_cluster_name
  db_api_url = var.db_api_url
}

output "llama_inference_deploy_function_url" {
  value = module.llama_inference_deploy.function_url
}

# provider "aws" {
#   region  = var.region
#   profile = var.awscli_profile
# }

terraform {
  backend "s3" {
    bucket  = "sskai-terraform-state"
    key     = "llama_inference_deploy/tf.state"
    region  = "ap-northeast-2"
    encrypt = true
  }
}
