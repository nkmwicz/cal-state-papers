from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from neo4j import GraphDatabase
from typing import TypedDict


load_dotenv()


class ResponseSchema(BaseModel):
    author: str
    auth_birth_year: int
    auth_death_year: int
    auth_location: str
    recipient: str
    recp_birth_year: int
    recp_death_year: int
    recp_location: str
    place_entities: list[str]
    people_entities: list[str]
    is_auth_in_db: bool
    is_recp_in_db: bool
    is_authloc_in_db: bool
    is_recploc_in_db: bool


class Person(TypedDict):
    name: str
    birthYear: int | None
    deathYear: int | None


def config_gemini():
    """
    Configures Gemini client and generation settings.

    Response Schema:
        - author: str
        - auth_birth_year: int
        - auth_death_year: int
        - auth_location: str
        - recipient: str
        - recp_birth_year: int
        - recp_death_year: int
        - recp_location: str
        - place_entitiies: list[str]
        - people_entities: list[str]
        - is_in_db: bool

    Returns:
        client: Configured Gemini client.
        config: Generation configuration with tools and response schema.
    """
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        system_instruction="You are an expert historian and researcher.",
        response_mime_type="application/json",
        response_schema=ResponseSchema,
    )
    return client, config


def get_db_people() -> list[Person]:
    """
    Connects to the Neo4j database and retrieves people nodes.
    Returns:
        List of dictionaries containing name, birthYear, and deathYear of people.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASS")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run(
            """MATCH (p:Person) RETURN p.name AS name, p.birthYear AS birthYear, p.deathYear AS deathYear"""
        )
        people: list[Person] = [
            {
                "name": record["name"],
                "birthYear": record["birthYear"],
                "deathYear": record["deathYear"],
            }
            for record in result
        ]

    driver.close()
    return people


def get_db_places() -> list[str]:
    """
    Connects to the Neo4j database and retrieves place nodes.
    Returns:
        List of place names.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASS")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run("""MATCH (pl:Location) RETURN pl.name AS name""")
        places = [record["name"] for record in result]

    driver.close()
    return places


def get_entities_from_text(
    text_content: str, db_people: list[dict], db_places: list[str], client, config
) -> dict:
    """
    Uses the Gemini API to extract entities from text, prioritizing a known list of people.

    Args:
        text_content (str): The historical text to analyze.
        db_people (list[dict]): A list of known people from the database.
        client: The configured Gemini client.
        config: The generation configuration for the client.

    Returns:
        dict: A dictionary containing the extracted entities, matching the ResponseSchema.
    """
    # Format the list of people for the prompt
    people_list_str = "\n".join(
        [
            f"- {p['name']} (Born: {p.get('birthYear', 'N/A')}, Died: {p.get('deathYear', 'N/A')})"
            for p in db_people
        ]
    )

    prompt = f"""
    You are an expert historian and researcher analyzing a summary of a historical document.
    Your task is to identify the author (sender) and the recipient (addressee) of the document.

    1.  **First, check this list of known individuals**:
        ```
        {people_list_str}
        ```
        If the author or recipient is in this list, use their information (name, birth year, death year) and set 'is_in_db' to true.

    2.  **If you cannot find a confident match in the list**, use your external knowledge and search capabilities to identify the author and recipient and their life dates. In this case, set 'is_in_db' to false.
    
    3. From the content of the text, determine the likely location associated with both the author and recipient. Use the following list of known places to assist you:
        ```
        {", ".join(db_places)}
        ```
    If you cannot confidently identify a location from the list, use your external knowledge to infer it.

    4.  Also, identify all other people and place names mentioned in the text following the same instructions as steps 1 and 2.

    5.  If you cannot identify an author or recipient, use the value "Unknown". Use 0 for unknown years.

    **Analyze the following text**:
    "{text_content}"

    Return your findings in the required JSON format.
    """

    try:
        response = client.generate_content(prompt, generation_config=config)
        return response.json()
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


def update_people_entities(
    people_entities: list[Person], new_entity: ResponseSchema
) -> list[Person]:
    """
    Cleans and deduplicates a list of people entities.

    class Person:
        name: str
        birthYear: int | None
        deathYear: int | None

    class ResponseSchema:
        author: Person
        recipient: Person
        location: str
        mentioned_people: list[Person]
        mentioned_places: list[str]

    Args:
        people_entities (list[Person]): List of people entities.
        new_entity (ResponseSchema): New entity data to add.
    Returns:
        list[Person]: Updated list of unique people entities.
    """
    if new_entity.author != "Unknown" and not any(
        p["name"] == new_entity.author for p in people_entities
    ):
        people_entities.append(
            {
                "name": new_entity.author,
                "birthYear": (
                    new_entity.auth_birth_year
                    if new_entity.auth_birth_year != 0
                    else None
                ),
                "deathYear": (
                    new_entity.auth_death_year
                    if new_entity.auth_death_year != 0
                    else None
                ),
            }
        )
    if new_entity.recipient != "Unknown" and not any(
        p["name"] == new_entity.recipient for p in people_entities
    ):
        people_entities.append(
            {
                "name": new_entity.recipient,
                "birthYear": (
                    new_entity.recp_birth_year
                    if new_entity.recp_birth_year != 0
                    else None
                ),
                "deathYear": (
                    new_entity.recp_death_year
                    if new_entity.recp_death_year != 0
                    else None
                ),
            }
        )
    return people_entities


def update_place_entities(
    place_entities: list[str], new_entity: ResponseSchema
) -> list[str]:
    """
    Cleans and deduplicates a list of place entities.

    class ResponseSchema:
        author: Person
        recipient: Person
        location: str
        people_entities: list[Person]
        place_entities: list[str]

    Args:
        place_entities (list[str]): List of place entities.
        new_entity (ResponseSchema): New entity data to add.

    Returns:
        list[str]: Updated list of unique place entities.
    """
    if (
        new_entity.auth_location != "Unknown"
        and new_entity.auth_location not in place_entities
    ):
        place_entities.append(new_entity.auth_location)
    if (
        new_entity.recp_location != "Unknown"
        and new_entity.recp_location not in place_entities
    ):
        place_entities.append(new_entity.recp_location)
    for place in new_entity.place_entitiies:
        if place not in place_entities:
            place_entities.append(place)
    return place_entities
