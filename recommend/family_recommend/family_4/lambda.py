from family_4 import get_family_4_for_inference
import json

def handler(event, context):
    try:
        body = json.loads(event["body"])
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = body.get('region')
    try:
        family = get_family_4_for_inference(region_name)
    except Exception as e:
        raise e

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'family': family,
        })
    }
    return response

# for test
if __name__ == "__main__":
    family = get_family_4_for_inference("us-east-1")
    print(family)