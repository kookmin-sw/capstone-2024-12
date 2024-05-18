import { 
  S3Client, 
  CreateMultipartUploadCommand,
  CompleteMultipartUploadCommand ,
  UploadPartCommand
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const Bucket = 'sskai-model-storage';
const client = new S3Client({ region: 'ap-northeast-2' });

export const handler = async (event) => {
  let body, statusCode = 200;
  const headers = {
    'Content-Type': "application/json"
  };
  let req = JSON.parse(event.body);
  const { upload_type, user_uid, uid, filename } = req;
  if (!upload_type || !user_uid || !uid || !filename)
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid Parameters (need 4 parameters)'
      }),
      headers
    };
  
  const isZip = /\.zip$/.test(filename);
  const isPython = /\.py$/.test(filename);

  if (!isZip && !isPython)
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid File Extension (need .zip or .py)'
      }),
      headers
    };

  if (!['model', 'data', 'code'].includes(upload_type))
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid upload_type (model, data or code)',
      }),
      headers
    };

  const Key = `${user_uid}/${upload_type}/${uid}/${isZip ? `${upload_type}.zip` : 'sskai_load_data.py' }`;
  
  try {
    switch (event.routeKey) {
      case 'POST /upload/start':
        statusCode = 201;
        const { UploadId } = await client.send(new CreateMultipartUploadCommand({
          Bucket,
          Key
        }));
        body = { UploadId };
        break;
      
      case 'POST /upload/url':
        statusCode = 201;
        body = await getSignedUrl(client, new UploadPartCommand({
          Bucket,
          Key,
          UploadId: req.UploadId,
          PartNumber: req.PartNumber
        }), { expiresIn: 600 });
        break;
      
      case 'POST /upload/complete':
        body = await client.send(new CompleteMultipartUploadCommand({
          Bucket,
          Key,
          UploadId: req.UploadId,
          MultipartUpload: {
            Parts: req.Parts
          }
        }));
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
