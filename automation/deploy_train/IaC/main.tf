module "train_deploy" {
  source                     = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix                     = "train_deploy"
  container_registry         = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository       = "train-deploy"
  container_image_tag        = "latest"
  lambda_ram_size            = 256
  attach_s3_policy           = true
  attach_ec2_policy          = true
  attach_eks_policy          = true
  attach_ssm_readonly_policy = true
  region_name = var.region
  eks_cluster_name = var.eks_cluster_name
  db_api_url = var.db_api_url
  upload_s3_url = var.upload_s3_url
}

output "train_deploy_function_url" {
  value = module.train_deploy.function_url
}

provider "aws" {
  region  = var.region
  profile = var.awscli_profile
}

terraform {
  backend "s3" {
    bucket  = "sskai-terraform-state"
    key     = "train_deploy/tf.state"
    region  = "ap-northeast-2"
    encrypt = true
  }
}
