import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
model = load_model('./bert_imdb')

def lambda_handler(event, context):
    json_body = json.loads(event['body'])
    input_ids = np.array(json_body['inputs']['input_ids'])
    input_masks = np.array(json_body['inputs']['input_masks'])
    segment_ids = np.array(json_body['inputs']['segment_ids'])
    result = model.predict([input_masks, input_ids, segment_ids])
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'body': result.tolist(),
        })
    }
    return response