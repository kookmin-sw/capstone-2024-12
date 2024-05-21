import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  QueryCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";
import { randomUUID } from 'crypto';

const region = process.env.AWS_REGION;
const client = new DynamoDBClient({ region });
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-trains";
const REQUIRED_FIELDS = ["name", "data", "user"];

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /trains":
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
            model: data.model,
            data: data.data,
            status: "Pending",
            cost: 0,
            checkpoint_path: "",
            epoch_num: data.epoch_num,
            learning_rate: data.learning_rate,
            optim_str: data.optim_str,
            loss_str: data.loss_str,
            train_split_size: data.train_split_size,
            batch_size: data.batch_size,
            worker_num: data.worker_num,
            data_loader_path: data.data_loader_path,
            class: data.class,
            model_type: data.model_type,
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Train created", train: command.Item };
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
            data_loader_path: data.data_loader_path || Item.data_loader_path,
            updated_at: new Date().getTime(),
            ...(data?.status === 'Running' && { start_at: new Date().getTime() }),
            ...(data?.status === 'Completed' && { end_at: new Date().getTime() }),
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
