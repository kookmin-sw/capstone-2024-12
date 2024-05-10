provider "aws" {
    region = var.region
    profile = var.awscli_profile
}

module "create_raycluster" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_api_template"
  prefix = "create_raycluster"
  container_registry = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
  container_repository = "create_raycluster"
  container_image_tag = "latest"
  lambda_ram_size = 1024
  attach_s3_policy = true
  attach_ec2_policy = true
  attach_eks_policy = true
  eks_cluster_name = "kuberay"
  karpenter_node_iam_node_name = "Karpenter-kuberay-20240510025703561000000029"
  # module.karpenter.karpenter_node_iam_node_name
}



terraform {
  backend "s3" {
    bucket = "sskai-terraform-state"
    key = "train_IaC/tf.state"
    region = "ap-northeast-2"
    encrypt = true
  }
}