output "function_url" {
  value = aws_lambda_function_url.lambda-url.function_url
}

output "function_name" {
  value = aws_lambda_function.lambda.function_name
}
