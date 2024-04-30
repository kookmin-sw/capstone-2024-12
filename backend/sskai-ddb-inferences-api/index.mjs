import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  ScanCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";
import { randomUUID } from 'crypto';

const client = new DynamoDBClient({});
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-inferences";
const REQUIRED_FIELDS = ["name", "model", "type", "endpoint"];

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /inferences":
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
            type: data.type,
            endpoint: data.endpoint,
            cost: 0,
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Inference created", inference: command };
        break;

        case "GET /inferences":
          command = { 
            TableName,
          };
          await dynamo.send(new ScanCommand(command));
          body = await dynamo.send(new ScanCommand(command));
          body = body.Items;
          break;

      case "GET /inferences/{id}":
        body = await dynamo.send(new GetCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
        }));
        body = body.Item;
        break;

      case "PUT /inferences/{id}":
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
            type: data.type || Item.type,
            endpoint: data.endpoint || Item.endpoint,
            cost: data.cost || Item.cost,
            updated_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Inference updated", inference: command.Item };
        break;

      case "DELETE /inferences/{id}":
        await dynamo.send(new DeleteCommand({
          TableName,
          Key: {
            uid: event.pathParameters.id,
          },
        }));
        body = { message: "Inference deleted", uid: event.pathParameters.id };
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
