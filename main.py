
# import extract_text
import extract_text_tesseract
# from langchain.llms import OpenAI
from kor.extraction import create_extraction_chain
# from langchain.chains import create_extraction_chain
from kor.nodes import Object, Text
from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
import json 
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from kor import from_pydantic
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    # texts = extract_text.detect_text_uri("https://tellerschophouse.com/wp-content/uploads/2021/07/tellers_menu_wineDict_07.02.21-1.jpg")
    
    filename = 'primeStam_menu_wineList_08.25.21-1.jpg'
    image_text = extract_text_tesseract.return_text(filename)
    with open(f'{filename}.txt', 'w') as f:
       f.write(image_text)
    with open(f'{filename}.txt') as f:
        read_text_file = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 600,
    chunk_overlap  = 50,
    length_function = len,
    add_start_index = True,
)

    texts = text_splitter.create_documents([read_text_file])
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=3000,
    model_kwargs={
        "frequency_penalty":0,
        "presence_penalty":0,
        "top_p":1.0,
        }
    )

    schema = Object(
        id="wine",
        description="wine product information",
        examples=[
            ("ZINFANDEL - MIRA FLORES (2018) EL DORADO GRANITE MINERALITY, RIPE POMEGRANATE AND SPICY RASPBERRY", 
             {"type": "white",
              "varietal": "zinfandel", 
              "producer": "mira flores", 
              "region": "el dorado", 
              "vintage": "2018"
              }
            ), 
            ('''F. Coppola Gold Label, Chardonnay 2008 34
                California''', 
             {"type": "white",
              "varietal": "chardonnay", 
              "producer": "f. coppola", 
              "state": "california",
              "country": "united states", 
              "vintage": "2008", 
              "price": "34",
              "product_name": "gold label"} 
              ),   
        ],
        attributes=[
             Text(
                id="type",
                description="The name of the color of the wine.",
            ),
            Text(
                id="varietal",
                description="The name of the grape varietal from which the wine is made.",
            ),
            Text(
                id="designation",
                description="The designation of the wine.",
            ),
             Text(
                id="producer",
                description="The name of the person or company who made the wine.",
            ),
            Text(
                id="product_name",
                description="The name of the wine product.",
            ),
            Text(
                id="state",
                description="The name of the state the wine was made in.",
            ),
            Text(
                id="city",
                description="The name of the city the wine was made in.",
            ),
            Text(
                id="country",
                description="The name of the country the wine was made in.",
            ),
            Text(
                id="region",
                description="The name of the place in which the wine was made.",
            ),
            Text(
                id="vintage",
                description="The year the wine was made.",
            ),
        ],
        many=True, 
    )

    # class Wine(BaseModel):

    #     type: Optional[str] = Field(
    #         description="the type of wine.", 
    #         examples=[
    #             ("My favorite types of wine are red, white, rose, orange, and sparkling",
    #             "red, white, rose, orange, sparkling"
    #             )
    #         ], 
    #     )
    #     varietal: Optional[str] = Field(
    #         description="The name of the grape varietal from which the wine is made.", 
    #         examples=[
    #             ("ZINFANDEL - MIRA FLORES (2018) EL DORADO GRANITE MINERALITY, RIPE POMEGRANATE AND SPICY RASPBERRY",
    #             "zinfandel"
    #             ),
    #             ("Schmitt Sohne Relax Reisling", "riesling")

    #         ], 
    #     )
    #     producer: Optional[str] = Field(
    #         description="The name of the person or company who made the wine.",
    #         examples=[ ("ZINFANDEL - MIRA FLORES (2018) EL DORADO GRANITE MINERALITY, RIPE POMEGRANATE AND SPICY RASPBERRY",
    #             "mira flores"
    #             ), 
    #             ("Veuve Cliquot Ponsardin Brut", "veuve cliquot"
    #             )],
    #     )
    #     region: Optional[str] = Field(
    #         description="The name of the place in which the wine was made.",
    #         examples=[
    #             ("Monterey, California", "monterey"),
    #             ("Champagne, France", "champagne"),
    #             ("Willamette Valley, Oregon", "willamette valley"),
    #             ("Lorie Valley, France", "loire valley"),
    #             ("Puglia, Italy", "puglia")
    #         ],
    #     )
    #     product_name: Optional[str] = Field(
    #         description="The name of the wine product.",
    #         examples=[
    #             ("Veuve Cliquot Ponsardin Brut",
    #             "ponsardin"
    #             ),
    #             ("Chateau Mouton Rothschild 2009 Pauillac, Premier Grand Cru",
    #             "pauillac"
    #             )],
    #     )
    #     designation: Optional[str] = Field(
    #         description="The designation of the wine product.",
    #         examples=[
    #             ("Veuve Cliquot Ponsardin Brut",
    #             "brut"
    #             ),
    #             ("Chateau Mouton Rothschild 2009 Pauillac, Premier Grand Cru",
    #             "premier grand cru"
    #             )],
    #     )
    #     vintage: Optional[str]
    #     country: Optional[str]
    #     state: Optional[str]
    #     city: Optional[str]
    #     price: Optional[str]
    # class WineList(BaseModel):
    #     wine: List[Wine]
    # schema, validator = from_pydantic(WineList) 
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")
    appended_results = []
    for i in range(len(texts)):
        results = chain.run(texts[i].page_content)['data']
        print(results)
        if results != {}:
            appended_results.extend(results['wine'])
    print (appended_results)
    json_object = json.dumps(appended_results, indent=4)
    with open(f'{filename}.json', 'w') as f:
         f.write(json_object)



if __name__ == "__main__":
    main()
