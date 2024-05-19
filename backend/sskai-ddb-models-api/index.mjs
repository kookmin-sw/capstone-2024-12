import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
  DeleteCommand,
  QueryCommand,
} from "@aws-sdk/lib-dynamodb";
import { DeleteObjectCommand, DeleteObjectsCommand, S3Client } from "@aws-sdk/client-s3";
import { randomUUID } from 'crypto';

const region = process.env.AWS_REGION;
const client = new DynamoDBClient({ region });
const clientS3 = new S3Client({ region });
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-models";
const Bucket = process.env.BUCKET_NAME;
const REQUIRED_FIELDS = ["name", "type", "user"];

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /models":
        if (!REQUIRED_FIELDS.every((field) => data[field])) {
          statusCode = 400;
          body = { message: "Missing required fields" };
          break;
        }
        statusCode = 201;
        command = {
          TableName,
          Item: {
            uid: randomUUID(),
            user: data.user,
            name: data.name,
            type: data.type,
            inferences: [],
            s3_url: data.s3_url,
            input_shape: data.input_shape,
            value_type: data.value_type,
            value_range: data.value_range,
            deploy_platform: data.deploy_platform,
            max_used_ram: data.max_used_ram,
            max_used_gpu_mem: data.max_used_gpu_mem,
            inference_time: data.inference_time,
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Model created", model: command.Item };
        break;

      case "GET /models":
        body = await dynamo.send(new QueryCommand({
          TableName,
          IndexName: "user-index",
          KeyConditionExpression: "#user = :user",
          FilterExpression: "attribute_exists(s3_url)",
          ExpressionAttributeNames: {
            "#user": "user"
          },
          ExpressionAttributeValues: {
            ":user": event.headers.user,
          },
        }));
        body = body.Items;
        break;

      case "GET /models/{id}":
        body = await dynamo.send(new GetCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
        }));
        body = body.Item;
        if (!body) {
          statusCode = 404;
          body = { message: "Not Found" };
        }
        break;

      case "PUT /models/{id}":
        const { Item } = await dynamo.send(new GetCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
        }));  
        if (!Item) {
          statusCode = 404;
          body = { message: "Not Found" };
          break;
        }
        command = {
          TableName,
          Item: {
            ...Item,
            name: data.name || Item.name,
            inferences: data.inferences || Item.inferences,
            s3_url: data.s3_url || Item.s3_url,
            deploy_platform: data.deploy_platform || Item.deploy_platform,
            max_used_ram: data.max_used_ram || Item.max_used_ram,
            max_used_gpu_mem: data.max_used_gpu_mem || Item.max_used_gpu_mem,
            inference_time: data.inference_time || Item.inference_time,
            input_shape: data.input_shape || Item.input_shape,
            value_type: data.value_type || Item.value_type,
            value_range: data.value_range || Item.value_range,
            updated_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Model updated", model: command.Item };
        break;

      case "DELETE /models/{id}":
        const deleted = await dynamo.send(new DeleteCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
          ReturnValues: "ALL_OLD",
        }));

        if (!deleted.Attributes) {
          statusCode = 404;
          body = { message: "Not Found" };
          break;
        }

        const model_url = `${deleted.Attributes.user}/model/${deleted.Attributes.uid}`

        const deleteFileCommand = new DeleteObjectsCommand({
          Bucket,
          Delete: {
            Objects: [{ Key: `${model_url}/model.zip` }, { Key: `${model_url}/model.tar.gz` }]
          }
        });

        const deletedDirCommand = new DeleteObjectCommand({
          Bucket,
          Key: `${model_url}/`,
        });

        await clientS3.send(deleteFileCommand);
        await clientS3.send(deletedDirCommand);

        body = { message: "Model deleted", uid: event.pathParameters.id, model: deleted.Attributes };
        break;
    }
  } catch (err) {
    statusCode = 400;
    body = err.message;
  } finally {
    body = JSON.stringify(body);
  }

  return { statusCode, body, headers };

};
