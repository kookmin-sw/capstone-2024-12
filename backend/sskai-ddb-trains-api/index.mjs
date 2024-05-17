import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  QueryCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";
import { randomUUID } from 'crypto';

const client = new DynamoDBClient({});
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-trains";
const REQUIRED_FIELDS = ["name", "data", "checkpoint_path", "desired_epoch", "user"];

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /trains":
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
            model: data.model,
            data: data.data,
            status: "Pending",
            cost: 0,
            checkpoint_path: "",
            desired_epoch: data.desired_epoch,
            code_path: data.code_path,
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Train created", model: command.Item };
        break;

      case "GET /trains":
        body = await dynamo.send(new QueryCommand({
          TableName,
          IndexName: "user-index",
          KeyConditionExpression: "#user = :user",
          ExpressionAttributeNames: {
            "#user": "user"
          },
          ExpressionAttributeValues: {
            ":user": event.headers.user,
          },
        }));
        body = body.Items;
        break;

      case "GET /trains/{id}":
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

      case "PUT /trains/{id}":
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
            model: data.model || Item.model,
            data: data.data || Item.data,
            status: data.status || Item.status,
            cost: data.cost || Item.cost,
            checkpoint_path: data.checkpoint_path || Item.checkpoint_path,
            updated_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Train updated", model: command.Item };
        break;

      case "DELETE /trains/{id}":
        await dynamo.send(new DeleteCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
        }));
        body = { message: "Train deleted", uid: event.pathParameters.id };
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
