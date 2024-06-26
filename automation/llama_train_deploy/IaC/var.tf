variable "region" {
  type    = string
  default = "ap-northeast-2"
}

variable "awscli_profile" {
  type    = string
  default = ""
}

variable "eks_cluster_name" {
    type = string
    default = ""
}

variable "db_api_url" {
    type = string
    default = "" 
}

variable "upload_s3_url" {
    type = string
    default = "" 
}

variable "container_registry" {
    type = string
    default = "609174368731.dkr.ecr.ap-northeast-2.amazonaws.com"
}