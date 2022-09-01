# yelp-cleaners
TDI project: predicting restaurant cleanliness with Yelp reviews

---

## motivation
As a Yelp user, I want to have a "dirtiness" number that I can associate with each restaurant page that can affect my decision to dine there. Normally, I do this by scrolling through reviews (and pictures) to see if there are any big red flags like "bugs in soup" or "unwiped tables." I want to make and train a model that can do this more systematically and automatically. Specifically, this model does NLP on user reviews, comparing against publicly available health inspection scores.

This model would ideally allow a user to input a restaurant name/address and spit out a cleanliness/dirtiness score along with some important snippets from review text (red/green flags).

---

## data and data ingestion
To limit the scope of the project, I look only at restaurants and inspection scores from the city of Austin, TX.

The review and business data comes from the 2021 Yelp Academic Dataset, and health inspection scores come from the city of Austin (https://data.austintexas.gov/Health-and-Community-Services/Food-Establishment-Inspection-Scores/ecmv-9xxi). Yelp prohibits scraping reviews from their website, and does not offer more review data through their API. The same is true with Google reviews.

Businesses from the Yelp and health inspection datasets were matched by querying the Yelp API. The cleaned dataset features **570614 reviews** for **2506 restaurants**, with **6617 total inspections**.

---

## strategy

The model has to be split into two parts:
1. Individual review model
2. Aggregation model

First, each individual review must be analyzed to get some measure of cleanliness. This should be treated as an **unsupervised** problem, as individual reviews cannot accurately predict the cleanliness of an overall business - there are too many reviews are irrelevant to restaurant cleanliness!

Second, the output of the individual review model must be aggregated to predict the cleanliness of each business. This is a **supervised** problem, as each business has an inspection score that we can train a model against. However, there aren't many data points here with only 2506 restaurants and 6617 inspections (can choose to only use reviews that occur within some time of an inspection).

---

## models

A simple bag of words on individual reviews results in an R^2 score of <0.1, showing some minor correlation.

