variable "main_suffix" {
  type = string
  default = ""
}

variable "region" {
  type = string
  default = ""
}

variable "awscli_profile" {
  type = string
  default = ""
}

variable "vpc_cidr" {
  type = string
  default = "192.168.0.0/16"
}
variable "public_subnet_cidrs" {
  description = "cidr should be match with public_subnet_number"
  type = list(string)
  default = ["192.168.10.0/24", "192.168.20.0/24", "192.168.30.0/24"]
}

variable "private_subnet_cidrs" {
  description = "cidr should be match with private_subnet_number"
  type = list(string)
  default = ["192.168.11.0/24", "192.168.21.0/24", "192.168.31.0/24"]
}

variable "cluster_name" {
  type = string
  default = "sskai-k8s-cluster"
}

variable "cluster_version" {
  type = string
  default = "1.29"
}