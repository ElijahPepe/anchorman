import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import praw

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID", None)
CLIENT_SECRET = os.getenv("CLIENT_SECRET", None)
PASSWORD = os.getenv("PASSWORD", None)
USERNAME = os.getenv("USERNAME", None)
SUBREDDIT = os.getenv("SUBREDDIT", None)
THRESHOLD = os.getenv("THRESHOLD", 0.65)

model = SentenceTransformer("all-MiniLM-L6-v2")

try:
	reddit = praw.Reddit(
    	client_secret = CLIENT_SECRET,
  	  client_id = CLIENT_ID,
    	password = PASSWORD,
    	username = USERNAME,
    	user_agent = "anchorman v1.0.0",
	)
	print("Logging in as " + str(reddit.user.me()))
except Exception as exception:
  print(str(exception))
  exit()

subreddit = reddit.subreddit(SUBREDDIT)

def compare_headlines(headline1, headline2):
	embeddings1 = model.encode(headline1, convert_to_tensor=True)
	embeddings2 = model.encode(headline2, convert_to_tensor=True)

	cosine_scores = util.cos_sim(embeddings1, embeddings2)

	return cosine_scores[0][0].item()

def get_recent_posts():
  return subreddit.new(limit=15)

for post in subreddit.stream.submissions(skip_existing=True):
  for submission in get_recent_posts():
    if submission != post:
      print(submission.title)
      print(post.title)
      score = compare_headlines(submission.title, post.title)
      if score > THRESHOLD:
        post.delete()