variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "prefix" {
  type    = string
  default = ""
}

variable "container_registry" {
  type    = string
  default = ""
}

variable "container_repository" {
  type    = string
  default = ""
}

variable "container_image_tag" {
  type    = string
  default = "latest"
}

variable "lambda_ram_size" {
  type    = number
  default = 4096
}
