variable "region" {
  type = string
  default = "us-east-1"
}

variable "awscli_profile" {
  type = string
  default = "default"
}

variable "db_api_url" {
  type = string
  default = "https://"
}

variable "inference_spot_api_url" {
  type = string
  default = "https://"
}

variable "inference_serverless_api_url" {
  type = string
  default = "https://"
}

variable "model_profile_api_url" {
  type = string
  default = "https://"
}

variable "user_train_api_url" {
  type = string
  default = "https://"
}

variable "streamlit_api_url" {
  type = string
  default = "https://"
}

variable "inference_llama_api_url" {
  type = string
  default = "https://"
}

variable "inference_diffusion_api_url" {
  type = string
  default = "https://"
}

variable "llama_train_api_url" {
  type = string
  default = "https://"
}

variable "diffusion_train_api_url" {
  type = string
  default = "https://"
}