import json
import logging
import os
import platform
import base64
import io

import cv2
import numpy as np
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field, validator
from pytesseract import pytesseract

from utils.aws import get_secret_dict

logger = logging.getLogger("recipe-crate")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if platform.system() == "Darwin":
    # Switch to the "personal" AWS profile
    os.environ["AWS_PROFILE"] = "personal"

secret_dict = get_secret_dict("prod/recipe-crate/openai-key")


class Recipe(BaseModel):
    title: str = Field(description="title of the recipe")
    blurb: str = Field(description="short description of the recipe")
    ingredients: list = Field(
        description="list of ingredients represented as dictionary objects with keys 'quantity', 'unit', 'ingredient', 'preparation"
    )
    method: list = Field(description="list of steps to make the recipe")
    other: list = Field(description="other information")

    def to_json(self):
        return json.dumps(
            {
                "title": self.title,
                "blurb": self.blurb,
                "ingredients": self.ingredients,
                "method": self.method,
                "other": self.other,
            },
            indent=4,
        )

    # # You can add custom validation logic easily with Pydantic.
    # @validator("setup")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field


def lambda_handler(event, context):
    # Get Image from event
    encoded_image = event["body"]["image"]

    # Decode image
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8)

    # Read Image
    path_to_tesseract = "/opt/homebrew/opt/tesseract/bin/tesseract"
    pytesseract.tesseract_cmd = path_to_tesseract

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- dilation on the green channel ---
    logger.info("Processing image")
    dilated_img = cv2.dilate(img[:, :, 1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    # --- finding absolute difference to preserve edges ---
    diff_img = 255 - cv2.absdiff(img[:, :, 1], bg_img)

    # --- normalizing between 0 to 255 ---
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    # --- Otsu threshold ---
    th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Extract Text
    logger.info("Extracting text")
    text = pytesseract.image_to_string(th)

    # Define Response Schema
    logger.info("Building Prompt")
    parser = PydanticOutputParser(pydantic_object=Recipe)
    format_instructions = parser.get_format_instructions()

    # Build Prompt Template
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""You are an assistant that parses raw text. 
            The text you are going to parse is a recipe read by an OCR. 
            The text data is very messy and needs to be structuered by you.
            You will parse the text into 5 parts:
            1. The title
            2. The ingredients
            3. The method
            4. The blurb -> a short description of the recipe
            5. Any other information -> for example how many it serves, or what page it's on, etc.

            {format_instructions}
            
            The text you are going to parse is: {text}""",
            input_variables=["text"],
            partial_variables={"format_instructions": format_instructions},
        )
    )
    # system_message_prompt = SystemMessagePromptTemplate(
    #     prompt=PromptTemplate(
    #         template="You are a helpful assistant that parses raw text",
    #         input_variables=[],
    #     )
    # )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            # system_message_prompt,
            human_message_prompt,
        ]
    )

    # Initialise AI
    chat = ChatOpenAI(
        temperature=0.7,
        openai_api_key=secret_dict["OPENAI_API_KEY"],
    )

    # Create Chai
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    # Run Chain
    logger.info("Querying AI")
    response = chain.run(text=text)

    # Parse Output
    logger.info("Parsing Output")
    try:
        parsed_response = parser.parse(response)
    except Exception as e:
        new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
        parsed_response = new_parser.parse(response)

    logger.info(f"Success! Output: {parsed_response.title}")

    return {"statusCode": 200, "body": parsed_response.to_json()}


if __name__ == "__main__":
    # Get all files im sample_data/recipes
    for filename in os.listdir("sample_data/recipes"):
        image_path = f"sample_data/recipes/{filename}"
        logger.info(f"Reading image: {image_path}")
        # Open image and convert to base64
        with open(image_path, "rb") as f:
            image = base64.b64encode(f.read())
        # Create event
        event = {"body": {"image": image}}
        # Run Lambda
        res = lambda_handler(event, None)

        with open(f"output/{filename}.json", "w") as f:
            logger.info(f"Writing output to: {f.name}")
            f.write(res["body"])
