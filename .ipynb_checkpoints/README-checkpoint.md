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

The 