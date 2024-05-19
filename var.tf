variable "region" {
  type = string
  default = "us-east-1"
}

variable "awscli_profile" {
  type = string
  default = "default"
}

variable "main_suffix" {
  type = string
  default = "sskai"
}

variable "container_registry" {
  type = string
  default = "694448341573.dkr.ecr.us-east-1.amazonaws.com"
}