import json
from utils.aws import get_secret_dict
import platform
import os

if platform.system() == "Darwin":
    # Switch to the "personal" AWS profile
    os.environ["AWS_PROFILE"] = "personal"

secret_dict = get_secret_dict("prod/recipe-crate/openai-key")

print(secret_dict)


def lambda_handler(event, context):
    print("Hello World!")
    return {"statusCode": 200, "body": json.dumps("Hello from Lambda!")}
