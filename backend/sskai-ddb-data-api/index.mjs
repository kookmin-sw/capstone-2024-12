import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  QueryCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";
import { DeleteObjectCommand, DeleteObjectsCommand, S3Client } from "@aws-sdk/client-s3";
import { randomUUID } from 'crypto';

const region = process.env.AWS_REGION;
const client = new DynamoDBClient({ region });
const clientS3 = new S3Client({ region });
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-data";
const Bucket = process.env.BUCKET_NAME;
const REQUIRED_FIELDS = ["name", "user"];

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /data":
        if (!REQUIRED_FIELDS.every((field) => data.hasOwnProperty(field) && data[field] != null)) {
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
            s3_url: data.s3_url,
            format: data.format,
            trains: [],
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Data created", data: command.Item };
        break;
      
      case "GET /data":
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

      case "GET /data/{id}":
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

      case "PUT /data/{id}":
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
            s3_url: data.s3_url || Item.s3_url,
            format: data.format || Item.format,
            trains: data.trains || Item.trains,
            updated_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Data updated", data: command.Item };
        break;

      case "DELETE /data/{id}":
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

        const data_url = `${deleted.Attributes.user}/data/${deleted.Attributes.uid}`

        const deleteFileCommand = new DeleteObjectsCommand({
          Bucket,
          Delete: {
            Objects: [{ Key: `${data_url}/data.zip` }, { Key: `${data_url}/data.tar.gz` }]
          }
        });

        const deletedDirCommand = new DeleteObjectCommand({
          Bucket,
          Key: `${data_url}/`,
        });

        await clientS3.send(deleteFileCommand);
        await clientS3.send(deletedDirCommand);

        body = { message: "Data deleted", uid: event.pathParameters.id, data: deleted.Attributes};
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
