resource "aws_dynamodb_table" "sskai-users" {
    name = "sskai-users"
    billing_mode = "PAY_PER_REQUEST"

    attribute {
        name = "uid"
        type = "S"
    }

    hash_key = "uid"
}

resource "aws_dynamodb_table" "sskai-models" {
    name = "sskai-models"
    billing_mode = "PAY_PER_REQUEST"

    attribute {
        name = "uid"
        type = "S"
    }

    attribute {
        name = "user"
        type = "S"
    }

    hash_key = "uid"

    global_secondary_index {
        name = "user-index"
        hash_key = "user"
        projection_type = "ALL"
    }
}

resource "aws_dynamodb_table" "sskai-trains" {
    name = "sskai-trains"
    billing_mode = "PAY_PER_REQUEST"

    attribute {
        name = "uid"
        type = "S"
    }

    attribute {
        name = "user"
        type = "S"
    }

    hash_key = "uid"

    global_secondary_index {
        name = "user-index"
        hash_key = "user"
        projection_type = "ALL"
    }
}

# resource "aws_dynamodb_table" "sskai-logs" {
#     name = "sskai-logs"
#     billing_mode = "PAY_PER_REQUEST"

#     attribute {
#         name = "uid"
#         type = "S"
#     }

    # attribute {
    #     name = "user"
    #     type = "S"
    # }

#     hash_key = "uid"

#     global_secondary_index {
#         name = "user-index"
#         hash_key = "user"
#         projection_type = "ALL"
#     }
# }

resource "aws_dynamodb_table" "sskai-inferences" {
    name = "sskai-inferences"
    billing_mode = "PAY_PER_REQUEST"

    attribute {
        name = "uid"
        type = "S"
    }

    attribute {
        name = "user"
        type = "S"
    }

    hash_key = "uid"

    global_secondary_index {
        name = "user-index"
        hash_key = "user"
        projection_type = "ALL"
    }
}

resource "aws_dynamodb_table" "sskai-data" {
    name = "sskai-data"
    billing_mode = "PAY_PER_REQUEST"

    attribute {
        name = "uid"
        type = "S"
    }

    attribute {
        name = "user"
        type = "S"
    }

    hash_key = "uid"

    global_secondary_index {
        name = "user-index"
        hash_key = "user"
        projection_type = "ALL"
    }
}