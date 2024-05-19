import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const region = process.env.AWS_REGION;
const Bucket = process.env.BUCKET_NAME;
const client = new S3Client({ region });

export const handler = async (event) => {
  const headers = {
    'Content-Type': "application/json"
  };
  
  const { upload_type, user_uid, uid, filename } = JSON.parse(event.body);
  
  if (!upload_type || !user_uid || !uid || !filename)
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid Parameters (need 3 parameters)'
      }),
      headers
    };

  const isZip = /\.zip$/.test(filename);

  if (!isZip)
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid File Extension (need .zip)'
      }),
      headers
    };
    
  if (!['model', 'data', 'data_loader'].includes(upload_type))
    return {
      statusCode: 400,
      body: JSON.stringify({
        message: 'Invalid upload_type (model, data or code)',
      }),
      headers
    };
  
  const command = new PutObjectCommand({ Bucket, Key: `${user_uid}/${upload_type}/${uid}/${upload_type}.zip` });
  const url = await getSignedUrl(client, command, { expiresIn: 3600 });
  
  return {
    statusCode: 201,
    body: JSON.stringify({
      message: 'Url Created',
      url
    }),
    headers
  };
};
