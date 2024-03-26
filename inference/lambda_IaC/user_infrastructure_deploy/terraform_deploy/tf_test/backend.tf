
    terraform {
      backend "s3" {
        bucket  = "skkai-lambda-test"
        key     = "user/user1/0c546fb407b523a09108cfbd51b7005f798484a15c0a64b8d39944e15207a80f/terraform.tfstate"
        region  = "ap-northeast-2"
        encrypt = true
      }
    }
    