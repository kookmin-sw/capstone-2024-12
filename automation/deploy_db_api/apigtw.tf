resource "aws_apigatewayv2_api" "sskai_api" {
  name = "SSKAI REST-API"
  protocol_type = "HTTP"
  cors_configuration {
    allow_methods = ["*"]
    allow_origins = ["*"]
    allow_headers = ["*"]
  }
}

### lambda permissions
resource "aws_lambda_permission" "sskai_ddb_data_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-ddb-data-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "sskai_ddb_inferences_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-ddb-inferences-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

# resource "aws_lambda_permission" "sskai_ddb_logs_api_permission" {
#   statement_id = "AllowAPIGatewayInvoke"
#   action = "lambda:InvokeFunction"
#   function_name = aws_lambda_function.sskai-ddb-logs-api.function_name
#   principal = "apigateway.amazonaws.com"
#   source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
# }

resource "aws_lambda_permission" "sskai_ddb_models_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-ddb-models-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "sskai_ddb_trains_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-ddb-trains-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "sskai_ddb_users_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-ddb-users-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "sskai_s3_multipart_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-s3-multipart-presigned-url.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "sskai_s3_api_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sskai-s3-presigned-url-api.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_apigatewayv2_api.sskai_api.execution_arn}/*/*"
}

### integrations
resource "aws_apigatewayv2_integration" "data_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-data-api.arn}/invocations"
    payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "inferences_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-inferences-api.arn}/invocations"
    payload_format_version = "2.0"
}

# resource "aws_apigatewayv2_integration" "logs_integration" {
#     api_id = aws_apigatewayv2_api.sskai_api.id
#     integration_type = "AWS_PROXY"
#     integration_method = "POST"
#     integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-logs-api.arn}/invocations"
#     payload_format_version = "2.0"
# }

resource "aws_apigatewayv2_integration" "models_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-models-api.arn}/invocations"
    payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "trains_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-trains-api.arn}/invocations"
    payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "users_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-ddb-users-api.arn}/invocations"
    payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "s3_multipart_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-s3-multipart-presigned-url.arn}/invocations"
    payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "s3_integration" {
    api_id = aws_apigatewayv2_api.sskai_api.id
    integration_type = "AWS_PROXY"
    integration_method = "POST"
    integration_uri = "arn:aws:apigateway:${var.region}:lambda:path/2015-03-31/functions/${aws_lambda_function.sskai-s3-presigned-url-api.arn}/invocations"
    payload_format_version = "2.0"
}

### routes
# /data
resource "aws_apigatewayv2_route" "data_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /data"
  target = "integrations/${aws_apigatewayv2_integration.data_integration.id}"
}

resource "aws_apigatewayv2_route" "data_get_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /data"
  target = "integrations/${aws_apigatewayv2_integration.data_integration.id}"
}

resource "aws_apigatewayv2_route" "data_put_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "PUT /data/{id}"
  target = "integrations/${aws_apigatewayv2_integration.data_integration.id}"
}

resource "aws_apigatewayv2_route" "data_get_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /data/{id}"
  target = "integrations/${aws_apigatewayv2_integration.data_integration.id}"
}

resource "aws_apigatewayv2_route" "data_delete_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "DELETE /data/{id}"
  target = "integrations/${aws_apigatewayv2_integration.data_integration.id}"
}

# /inferences
resource "aws_apigatewayv2_route" "inferences_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /inferences"
  target = "integrations/${aws_apigatewayv2_integration.inferences_integration.id}"
}

resource "aws_apigatewayv2_route" "inferences_get_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /inferences"
  target = "integrations/${aws_apigatewayv2_integration.inferences_integration.id}"
}

resource "aws_apigatewayv2_route" "inferences_put_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "PUT /inferences/{id}"
  target = "integrations/${aws_apigatewayv2_integration.inferences_integration.id}"
}

resource "aws_apigatewayv2_route" "inferences_get_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /inferences/{id}"
  target = "integrations/${aws_apigatewayv2_integration.inferences_integration.id}"
}

resource "aws_apigatewayv2_route" "inferences_delete_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "DELETE /inferences/{id}"
  target = "integrations/${aws_apigatewayv2_integration.inferences_integration.id}"
}

# # /logs
# resource "aws_apigatewayv2_route" "logs_post_route" {
#   api_id = aws_apigatewayv2_api.sskai_api.id
#   route_key = "POST /logs"
#   target = "integrations/${aws_apigatewayv2_integration.logs_integration.id}"
# }

# resource "aws_apigatewayv2_route" "logs_get_route" {
#   api_id = aws_apigatewayv2_api.sskai_api.id
#   route_key = "GET /logs"
#   target = "integrations/${aws_apigatewayv2_integration.logs_integration.id}"
# }

# resource "aws_apigatewayv2_route" "logs_put_id_route" {
#   api_id = aws_apigatewayv2_api.sskai_api.id
#   route_key = "PUT /logs/{id}"
#   target = "integrations/${aws_apigatewayv2_integration.logs_integration.id}"
# }

# resource "aws_apigatewayv2_route" "logs_get_id_route" {
#   api_id = aws_apigatewayv2_api.sskai_api.id
#   route_key = "GET /logs/{id}"
#   target = "integrations/${aws_apigatewayv2_integration.logs_integration.id}"
# }

# resource "aws_apigatewayv2_route" "logs_delete_id_route" {
#   api_id = aws_apigatewayv2_api.sskai_api.id
#   route_key = "DELETE /logs/{id}"
#   target = "integrations/${aws_apigatewayv2_integration.logs_integration.id}"
# }

# /models
resource "aws_apigatewayv2_route" "models_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /models"
  target = "integrations/${aws_apigatewayv2_integration.models_integration.id}"
}

resource "aws_apigatewayv2_route" "models_get_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /models"
  target = "integrations/${aws_apigatewayv2_integration.models_integration.id}"
}

resource "aws_apigatewayv2_route" "models_put_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "PUT /models/{id}"
  target = "integrations/${aws_apigatewayv2_integration.models_integration.id}"
}

resource "aws_apigatewayv2_route" "models_get_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /models/{id}"
  target = "integrations/${aws_apigatewayv2_integration.models_integration.id}"
}

resource "aws_apigatewayv2_route" "models_delete_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "DELETE /models/{id}"
  target = "integrations/${aws_apigatewayv2_integration.models_integration.id}"
}

# /trains
resource "aws_apigatewayv2_route" "trains_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /trains"
  target = "integrations/${aws_apigatewayv2_integration.trains_integration.id}"
}

resource "aws_apigatewayv2_route" "trains_get_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /trains"
  target = "integrations/${aws_apigatewayv2_integration.trains_integration.id}"
}

resource "aws_apigatewayv2_route" "trains_put_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "PUT /trains/{id}"
  target = "integrations/${aws_apigatewayv2_integration.trains_integration.id}"
}

resource "aws_apigatewayv2_route" "trains_get_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /trains/{id}"
  target = "integrations/${aws_apigatewayv2_integration.trains_integration.id}"
}

resource "aws_apigatewayv2_route" "trains_delete_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "DELETE /trains/{id}"
  target = "integrations/${aws_apigatewayv2_integration.trains_integration.id}"
}

# /users
resource "aws_apigatewayv2_route" "users_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /users"
  target = "integrations/${aws_apigatewayv2_integration.users_integration.id}"
}

resource "aws_apigatewayv2_route" "users_get_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /users"
  target = "integrations/${aws_apigatewayv2_integration.users_integration.id}"
}

resource "aws_apigatewayv2_route" "users_put_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "PUT /users/{id}"
  target = "integrations/${aws_apigatewayv2_integration.users_integration.id}"
}

resource "aws_apigatewayv2_route" "users_get_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "GET /users/{id}"
  target = "integrations/${aws_apigatewayv2_integration.users_integration.id}"
}

resource "aws_apigatewayv2_route" "users_delete_id_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "DELETE /users/{id}"
  target = "integrations/${aws_apigatewayv2_integration.users_integration.id}"
}

# /upload
resource "aws_apigatewayv2_route" "upload_post_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /upload"
  target = "integrations/${aws_apigatewayv2_integration.s3_integration.id}"
}

resource "aws_apigatewayv2_route" "upload_start_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /upload/start"
  target = "integrations/${aws_apigatewayv2_integration.s3_multipart_integration.id}"
}

resource "aws_apigatewayv2_route" "upload_url_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /upload/url"
  target = "integrations/${aws_apigatewayv2_integration.s3_multipart_integration.id}"
}

resource "aws_apigatewayv2_route" "upload_complete_route" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  route_key = "POST /upload/complete"
  target = "integrations/${aws_apigatewayv2_integration.s3_multipart_integration.id}"
}

resource "aws_apigatewayv2_stage" "apigtw_stage" {
  api_id = aws_apigatewayv2_api.sskai_api.id
  name = "sskai-api-dev"
  auto_deploy = true
}