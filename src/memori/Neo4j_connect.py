from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

# Avoid an expensive schema refresh during import/startup.
Neo4jGraph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    refresh_schema=False,
)
