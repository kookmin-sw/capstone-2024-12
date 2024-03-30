from recommender_inference_cpu_family import get_cpu_instance_family_for_inference
import json

def handler(event, context):
    try:
        param = event['queryStringParameters']
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = param['region']
    try:
        family = get_cpu_instance_family_for_inference(region_name)
    except Exception as e:
        raise e

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'family': family,
        })
    }
    return response