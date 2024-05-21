variable "api_list" {
  type = list(string)
  default = ["sskai-ddb-data-api",
    "sskai-ddb-inferences-api",
    "sskai-ddb-logs-api",
    "sskai-ddb-models-api",
    "sskai-ddb-trains-api",
    "sskai-ddb-users-api",
    "sskai-s3-multipart-presigned-url",
  "sskai-s3-presigned-url-api",
  "sskai-cost-calculate"]
}

variable "region" {
  type = string
  default = "us-east-1"
}

variable "awscli_profile" {
  type = string
  default = "default"
}