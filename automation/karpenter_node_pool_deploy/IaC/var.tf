variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "awscli_profile" {
  type    = string
  default = "mhsong-swj"
}

variable "eks_cluster_name" {
    type = string
    default = "swj-eks-test"
}