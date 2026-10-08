from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("mongodb_URI")

# Connect to MongoDB
client = MongoClient(connection_string)

# Select/Create database
db = client["memories"]