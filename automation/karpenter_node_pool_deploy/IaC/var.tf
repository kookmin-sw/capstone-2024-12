variable "region" {
  type    = string
  default = ""
}

variable "awscli_profile" {
  type    = string
  default = ""
}

variable "eks_cluster_name" {
    type = string
    default = ""
}

variable "container_registry" {
  type    = string
  default = "694448341573.dkr.ecr.ap-northeast-2.amazonaws.com"
}