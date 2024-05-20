module "serverless_inference" {
  source = "github.com/kookmin-sw/capstone-2024-12//IaC/serverless_inference"
  prefix = var.prefix
  container_registry = var.container_registry
  container_repository = "serverless-inference"
  container_image_tag = "latest"
  lambda_ram_size = var.lambda_ram_size
  model_s3_url = var.model_s3_url
  region = var.region
}

variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "prefix" {
  type = string
  default = ""
}
variable "container_registry" {
  type = string
  default = ""
}
variable "lambda_ram_size" {
  type = number
  default = 3072
}
variable "model_s3_url" {
  type = string
  default = ""
}

output "function_url" {
  value = module.serverless_inference.function_url
}

provider "aws" {
    region = var.region
}