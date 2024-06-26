import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  PutCommand,
  QueryCommand,
} from "@aws-sdk/lib-dynamodb";
import { randomUUID } from 'crypto';

const region = process.env.AWS_REGION;
const client = new DynamoDBClient({ region });
const dynamo = DynamoDBDocumentClient.from(client);
const TableName = "sskai-logs"

export const handler = async (event) => {
  let body, command, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    switch (event.routeKey) {
      case "POST /logs":
        statusCode = 201;
        command = {
          TableName,
          Item: {
            uid: randomUUID(),
            user: data.user,
            kind_of_job: data.kind_of_job,
            job: data.job,
            name: data.name,
            created_at: new Date().getTime(),
          }
        };
        await dynamo.send(new PutCommand(command));
        body = { message: "Log created", log: command };
        break;

      case "GET /logs":
        body = await dynamo.send(new QueryCommand({ 
          TableName,
          IndexName: "user-index",
          KeyConditionExpression: "#user = :user",
          ExpressionAttributeNames: {
            "#user": "user"
          },
          ExpressionAttributeValues: {
            ':user': event.headers.user,
          },
        }));
        body = body.Items;
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
