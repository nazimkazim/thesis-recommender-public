from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import time
import uuid


app = Flask(__name__)

# Load the trained model
cosine_similarities = load('model.joblib')
indices = load('indices.joblib')
# this assumes you've also saved your DataFrame with joblib
df = load('df.joblib')


def get_recommendations(title):
    if title not in indices:
        return f"No place titled '{title}' found in our database."
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_similarities[idx]))
    # inspect the values
    # print(f'Cosine Similarities: {cosine_similarities[idx]}')
    # print(f'Sim Scores: {sim_scores}')  # inspect the values before sorting
    # Sort the movies based on the similarity scores
    try:
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f'Error while sorting: {e}')
        # inspect the values when error happens
        # print(f'Sim Scores: {sim_scores}')
        return []
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return df.loc[movie_indices, ['name', 'business_id']].to_dict(orient='records')


@app.route('/predict', methods=['POST'])
def predict():
    request_id = uuid.uuid4()
    start_time = time.time()
    # Get the movie title from the POST request
    data = request.get_json(force=True)
    title = data['title']

    # Get the top 10 recommendations
    top_10 = get_recommendations(title)

    end_time = time.time()  # End time after processing the request
    elapsed_time = end_time - start_time  # Calculating the elapsed time

    # Logging the elapsed time
    print(
        f"Request ID: {request_id}, Time taken to generate recommendations: {elapsed_time:.4f} seconds")

    # Return the results
    return jsonify(top_10)


@app.route('/cpu-intensive', methods=['GET'])
def cpu_intensive():
    # default to 10000000 if not provided
    iterations = int(request.args.get('iterations', 10000000))
    start_time = time.time()
    result = 0
    for _ in range(0, iterations):
        result += 1
    duration = time.time() - start_time
    return jsonify({"result": result, "duration": duration})


@app.route('/health', methods=['GET'])
def health_check():
    return "Healthy", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
