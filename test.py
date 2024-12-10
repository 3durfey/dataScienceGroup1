from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Test by printing an environment variable
print(os.getenv('B2_BUCKETNAME'))