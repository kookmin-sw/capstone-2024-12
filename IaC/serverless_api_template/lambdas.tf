module "lambda" {
  source          = "./lambda"
  prefix          = var.prefix
  container_registry = var.container_registry
  container_repository = var.container_repository
  container_image_tag = var.container_image_tag
  ram_mib         = var.lambda_ram_size
}
