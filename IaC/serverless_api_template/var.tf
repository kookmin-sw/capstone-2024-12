variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "awscli_profile" {
  type    = string
  default = ""
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
  default = 2048
}

variable "attach_ec2_policy" {
  type = bool
  default = false
}

variable "attach_s3_policy" {
  type = bool
  default = false
}

variable "attach_vpc_policy" {
  type = bool
  default = false
}

variable "attach_lambda_policy" {
  type = bool
  default = false
}

variable "attach_cloudwatch_policy" {
  type = bool
  default = false
}