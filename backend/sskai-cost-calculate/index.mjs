import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { SSMClient, GetParameterCommand } from "@aws-sdk/client-ssm";
import {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
} from "@aws-sdk/lib-dynamodb";

const region = process.env.AWS_REGION;
const ssmClient = new SSMClient({ region });
const ddbClient = new DynamoDBClient({ region });
const dynamo = DynamoDBDocumentClient.from(ddbClient);
const TrainTable = "sskai-trains";
const InferenceTable = "sskai-inferences";

const REQUIRED_FIELDS = ["type", "uid", "platform", "ram"];
const ONDEMAND_MAPPER = {
  'nodepool-1': 'nodepool_1_ondemand_price',
  'nodepool-2': 'nodepool_2_ondemand_price',
  'nodepool-3': 'nodepool_3_ondemand_price',
  'nodepool-4': 'nodepool_4_ondemand_price',
  'nodepool-5': 'nodepool_5_ondemand_price',
}
const SPOT_MAPPER = {
  'nodepool-1': 'nodepool_1_spot_price',
  'nodepool-2': 'nodepool_2_spot_price',
  'nodepool-3': 'nodepool_3_spot_price',
  'nodepool-4': 'nodepool_4_spot_price',
  'nodepool-5': 'nodepool_5_spot_price',
}


export const handler = async (event) => {
  let body, statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
  };

  try {
    const data = JSON.parse(event.body || "{}");
    if (!REQUIRED_FIELDS.every((field) => data.hasOwnProperty(field) && data[field] != null)) {
      statusCode = 400;
      body = { message: "Missing required fields" };
    }
    
    const { type, uid, platform, ram } = data;
    const { Item } = await dynamo.send(new GetCommand({
      TableName: type === 'train' ? TrainTable : InferenceTable,
      Key: {
        uid: uid,
      },
    }));

    if (!Item) {
      statusCode = 404;
      body = { message: "Not Found" };
      throw Error;
    }

    let command;

    if (platform === 'Serverless') {
      command = {
        TableName: type === 'train' ? TrainTable : InferenceTable,
        Item: {
          ...Item,
          cost: Number(ram) * 0.000000016406,
          original_cost: Number(ram) * 0.000000002821181 + 0.000000771604938,
        }
      }
    } else {
      const ondemandName = ONDEMAND_MAPPER[platform];
      const spotName = SPOT_MAPPER[platform];
      
      const ondemandCost = await ssmClient.send(new GetParameterCommand({
        Name: ondemandName,
        WithDecryption: true,
      }));

      const spotCost = await ssmClient.send(new GetParameterCommand({
        Name: spotName,
        WithDecryption: true,
      }));
      
      command = {
        TableName: type === 'train' ? TrainTable : InferenceTable,
        Item: {
          ...Item,
          cost: Number(spotCost.Parameter.Value),
          original_cost: Number(ondemandCost.Parameter.Value),
        }
      };
    }
    await dynamo.send(new PutCommand(command));
    body = { message: "Cost calculated", [type]: command.Item };
  } catch (error) {
    if (body) console.error(body.message);
    else {
      statusCode = 500;
      console.error("Error:", error);
    }
  } finally {
    body = JSON.stringify(body);
  }

  return { headers, statusCode, body };
};
