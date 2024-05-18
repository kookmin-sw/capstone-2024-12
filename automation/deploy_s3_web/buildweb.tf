resource "local_file" "react_env_file" {
    filename = "${path.module}/../../frontend/sskai-console/.env"
    content = <<EOF
VITE_DB_API_URL= "${var.db_api_url}"
VITE_TMP_USER_UID = '5d9b890e-1316-4e25-8f67-829702a24331'
VITE_INFERENCE_SPOT_API_URL = "${var.inference_spot_api_url}"
VITE_INFERENCE_SERVERLESS_API_URL = "${var.inference_serverless_api_url}"
VITE_MODEL_PROFILE_API_URL = "${var.model_profile_api_url}"
VITE_USER_TRAIN_API_URL = "${var.model_profile_api_url}"
VITE_STREAMLIT_API_URL = "${var.stream_api_url}"
EOF
}

resource "null_resource" "build_react_app" {
  provisioner "local-exec" {
    command = "yarn && yarn build && aws s3 sync ./dist s3://sskai-s3-web-${random_id.random_string.hex}/ --profile ${var.awscli_profile}"
    working_dir = "${path.module}/../../frontend/sskai-console"
  }

  depends_on = [ aws_s3_bucket.sskai-s3-web-bucket ]
}