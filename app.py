import base64
import io
import json
import logging
import os
import platform

import numpy as np
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field, validator
from supabase import Client, create_client

from utils.aws import get_secret_dict

logger = logging.getLogger("recipe-crate")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if platform.system() == "Darwin":
    # Switch to the "personal" AWS profile
    os.environ["AWS_PROFILE"] = "personal"

secret_dict = get_secret_dict("prod/recipe-crate/openai-key")
SUPABASE_URL = secret_dict["SUPABASE_URL"]
SUPABASE_KEY = secret_dict["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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
    logger.info("Starting Lambda")
    logger.info(event)
    # Get text from event
    text = event["body"]["text"]

    # Define Response Schema
    logger.info("Building Prompt")
    parser = PydanticOutputParser(pydantic_object=Recipe)
    format_instructions = parser.get_format_instructions()

    # Build Prompt Template
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""You are an assistant that parses raw text. 
            The text you are going to parse is a recipe read by an OCR. 
            The text data is very messy and needs to be structured by you.
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

    print(chat_prompt_template.format(text=text))
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
    text = """Cinnamon & cherry tomato koshari
SERVES 6
½ cup green/brown lentils (150g)
750g fresh cherry tomatoes or 2 × 400g tins cherry tomatoes, drained
4 cloves of garlic
6 shallots, very thinly sliced
5 tablespoons extra virgin olive oil
a small bunch of coriander,
including stalks
1 teaspoon ground allspice or
4 allspice berries, bashed
1 stick of cinnamon
1 cup of rice (300g), rinsed rapeseed oil
a small bunch of parsley, roughly chopped a small bunch of mint, roughly chopped
This bravely spiced rice and lentil dish is more than a sum of its parts and comes (mostly) from the store cupboard. It's topped with a crown of crispy frizzled shallots, which brings a key level of crunch and contrast to what is otherwise soft and comforting. In its pure Egyptian street-food form there is pasta, macaroni to be precise, and often chickpeas, too.
I've just used rice and lentils here for simplicity, but you could add a handful of macaroni to the rice, or a drained can of chickpeas, to make
it a little more authentic.
First, soak your lentils in 300ml of warm water for an hour, then drain and rinse. Preheat the oven to 200°C/180°C fan/gas 6.
In a large cast-iron pot with a lid, put the tomatoes, garlic, a quarter of the sliced shallots and 2 tablespoons of the olive oil. Chop the coriander stalks then add those as well, along with the allspice and stick of cinnamon.
Roast in the oven for 10 minutes then remove from the oven and sprinkle on the rice and rinsed lentils. Add 900ml of boiling water, season with salt, cover with the lid and place in the oven for another 25 minutes.
Meanwhile, make the crispy shallots. Fill a saucepan with a 3cm depth of room-temperature rapeseed oil. Put over a high heat and then, once hot, add the remaining shallots. You should see the oil bubble up when you add them. Give them a quick stir so they are all evenly submerged in the oil, then leave to cook for around 4 minutes, keeping an eye on them as they cook. Once they start turning a light golden colour remove them from the oil with a slotted spoon and onto some kitchen paper then leave to cool completely.
Remove the rice from the oven and, using an oven glove, take off the lid and remove the whole garlic cloves. Let the rice stand for 10 minutes and, meanwhile, pop the garlic out of its skin and mash with some sea salt and the remaining 3 tablespoons extra virgin olive oil.
Stir the garlic oil through the rice followed by the chopped parsley, mint and coriander leaves. Season with salt and pepper and top with the crispy shallots.
42
ONE • POT"""
    event = {"body": {"text": text}}

    # Run Lambda
    res = lambda_handler(event, None)

    print(json.dumps(json.loads(res["body"]), indent=4))
    exit()

    recipe = json.loads(res["body"])

    logger.info(f"Writing output to database...")
    data, count = supabase.table("recipes").insert(recipe).execute()

    print(data)
