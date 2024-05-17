module "karpenter_nodepool_manager" {
  source                             = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix                             = "karpenter_nodepool_manager"
  container_registry                 = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository               = "deploy-karpenter-node-pool"
  container_image_tag                = "latest"
  lambda_ram_size                    = 256
  lambda_timeout                     = 600
  attach_s3_policy                   = true
  attach_ec2_policy                  = true
  attach_eks_policy                  = true
  attach_ssm_readonly_policy         = true
  region_name                        = "ap-northeast-2"
  karpenter_node_role_parameter_name = "karpenter_node_role_name_${var.eks_cluster_name}"

  eks_cluster_name = var.eks_cluster_name
}

output "karpenter_nodepool_manager_function_url" {
  value = module.karpenter_nodepool_manager.function_url
}

provider "aws" {
  region  = var.region
  profile = var.awscli_profile
}

terraform {
  backend "s3" {
    bucket  = "sskai-terraform-state"
    key     = "karpenter_manager/tf.state"
    region  = "ap-northeast-2"
    encrypt = true
  }
}
