provider "aws" {
  region = var.region
  profile = var.awscli_profile
}

provider "random" {}

resource "random_id" "random_string" {
  byte_length = 8
}

resource "aws_s3_bucket" "tfstate_bucket" {
  bucket = "sskai-terraform-state-${random_id.random_string.hex}"
  force_destroy = true
}

module "kubernetes_cluster" {
  source = "./IaC/kubernetes_cluster"
  main_suffix = var.main_suffix
  awscli_profile = var.awscli_profile
  region = var.region 
}

module "deploy_streamlit" {
  source = "./automation/deploy_streamlit/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api]
}

module "deploy_train" {
  source = "./automation/deploy_train/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  upload_s3_url = "${module.deploy_db_api.api_endpoint_url}/upload"
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api]
}

module "family_recommend" {
  source = "./recommend/family_recommend/IaC"
  container_registry = var.container_registry
  awscli_profile = var.awscli_profile
  region = var.region
}

module "karpenter_node_pool_deploy" {
  source = "./automation/karpenter_node_pool_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster ]
}

module "kubernetes_inference_deploy" {
  source = "./automation/kubernetes_inference_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api ]
}

module "kubernetes_model_profiler_deploy" {
  source = "./automation/kubernetes_model_profiler_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api ]
}

module "serverless_inference_deploy" {
  source = "./automation/serverless_inference_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  db_api_url = module.deploy_db_api.api_endpoint_url
  state_bucket_name = aws_s3_bucket.tfstate_bucket.bucket
  container_registry = var.container_registry

  depends_on = [ module.deploy_db_api ]
}

module "llama_inference_deploy" {
  source = "./automation/llama_inference_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api ]
}

module "diffusion_inference_deploy" {
  source = "./automation/diffusion_inference_deploy/IaC"
  awscli_profile = var.awscli_profile
  region = var.region
  eks_cluster_name = module.kubernetes_cluster.cluster_name
  db_api_url = module.deploy_db_api.api_endpoint_url
  container_registry = var.container_registry

  depends_on = [ module.kubernetes_cluster, module.deploy_db_api ]
}

module "deploy_db_api" {
  source = "./automation/deploy_db_api"
  awscli_profile = var.awscli_profile
  region = var.region
}

module "deploy_s3_web" {
  source = "./automation/deploy_s3_web"
  awscli_profile = var.awscli_profile
  region = var.region
  db_api_url = module.deploy_db_api.api_endpoint_url
  # diffusion_train_api_url = 
  # llama_train_api_url =
  inference_diffusion_api_url = module.diffusion_inference_deploy.diffusion_inference_deploy_function_url
  inference_llama_api_url = module.llama_inference_deploy.llama_inference_deploy_function_url
  streamlit_api_url = module.deploy_streamlit.streamlit_function_url
  user_train_api_url = module.deploy_train.train_deploy_function_url
  model_profile_api_url = module.kubernetes_model_profiler_deploy.kubernetes_model_profiler_deploy_function_url
  inference_serverless_api_url = module.serverless_inference_deploy.function_url
  inference_spot_api_url = module.kubernetes_inference_deploy.kubernetes_inference_deploy_function_url

  depends_on = [ module.deploy_db_api ]
}

resource "aws_lambda_invocation" "nodepool_deploy" {
  function_name = module.karpenter_node_pool_deploy.karpenter_nodepool_manager_function_name
  input = jsonencode({
    "a": "b"
  })
}