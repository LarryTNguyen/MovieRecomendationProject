# MovieRecomendationProject

<h2>Purpose of the project</h2>

The reason of this project is two-fold. Reason #1 was the fact that I really enjoy movies and always find myself trying to figure out what to watch whenever I do find the time to watch one. Reason #2 is that I have a lot of time on my hands right now and I wanted to improve my coding skills. As I work on this project from scratch, I hope I learn more about implementing a machine learning program with a front and back end.

# My journey so far

<h2> The Recommendation Algorithm </h2>

<h4>K-nearest Neighbors</h4>
My first part of the project was trying to figure out which model to use for the recommendation algorithm. At first, I used K-nearest neighbors as the model to recommend movies as it finds the movies that are closest to what the user has watched before. However, a flaw with that is that it doesn't really help the user venture out to different types of movies. Instead, it keeps the user to very similar movies that they have been watching. It also doesn't take into account if the user actually liked the movie they just watched. With these flaws in mind, I decided to look for a different model.

<h4>Singular Value Decomposition</h4>
The next model I tried to look at was Singular Value Decomposition with Matrix Factorization. With this method, we can use the user's ratings on previous movies to see what kind of movies they should be recommended. Working with this method was interesting as it was my first time working with matrix factorization so there was a learning curve attached to it. When things finally got to working, it seemed to work with the test users in the ratings dataset. However, when I tried to implement my own data into the algorithm, I got some very interesting results that I didn't think I would like. Instead of getting actual movies, I got a lot of obscure mini TV series that seemingly had no correlation with my actual tastes. Because of this skewed user data, I wanted to see the results through a different model.

<h4>LightFM</h4>
The current model I am working on right now is the LightFM model. It will take into both the user's ratings and the genre of movies they watched. This combines the content filtering and collaborative filtering within the two recommendation systems that we previously talked about. It is still in the works right now so I have yet to see what the consensus is with LightFM.
