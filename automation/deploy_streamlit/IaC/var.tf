variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "awscli_profile" {
  type    = string
  default = "kkmcapstone"
}

variable "eks_cluster_name" {
  type = string
  default = "kuberay"
}

variable "db_api_url" {
  type = string
  default = "https://wpcwvjlvkl.execute-api.ap-northeast-2.amazonaws.com/sskai-api-dev"
}