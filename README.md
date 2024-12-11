[Link to Site]((https://datasciencegroup1.streamlit.app))

### Abstract or Overview

The project is a data-driven web application designed to assist users in exploring rental property options based on their preferences. It integrates data filtering, scoring, machine learning predictions, and recommendation systems to create a comprehensive solution for rental property searches. Users can filter apartments by attributes such as the number of bedrooms, bathrooms, and price range. They can also assign weights to these criteria to prioritize results. 

This application is valuable for potential renters, real estate agents, and property managers, offering a streamlined platform to evaluate properties based on personalized preferences.

---

### Data Description

The application uses a dataset of rental properties (`10k_data.pickle`) containing attributes such as:

- Property details: Title, description, and address.
- Attributes: Number of bedrooms, bathrooms, square footage, price, and amenities.
- Location data: City, state, latitude, and longitude.
- Additional information: Pet policies, photo availability, and source.

The dataset was preprocessed to clean and validate entries:

- Removed duplicate and invalid data.
- Standardized columns (e.g., handling missing values and creating additional columns like `half_bathrooms`).
- Ensured proper formatting of location fields.

---

### Algorithm Description

The application is driven by multiple algorithms:

1. **Data Filtering**:
   - Filters apartments based on user-selected criteria, such as state, price range, and ratings for bathrooms and bedrooms.
2. **Scoring System**:
   - Scores apartments based on weighted criteria using a custom scoring algorithm (`ScoreDistribution`).
   - Sorts properties based on their relevance to user preferences.
3. **Machine Learning Prediction**:
   - A Random Forest Regressor predicts rental prices based on attributes such as square footage, number of bedrooms and bathrooms, and city.
4. **Recommendations**:
   - Utilizes PCA and pairwise distance calculations to find similar apartments based on top-ranked properties.
5. **Search Functionality**:
   - Uses text-based search (`Simple_Search`) to match user queries with apartment details, leveraging cosine similarity for relevance scoring.

---

### Tools Used

1. **Streamlit**:
   - Builds the web interface for user interaction.
   - Displays apartment details, visualizations, and predictions.
2. **Backblaze**:
   - Stores and retrieves large data files and machine learning models.
3. **Pandas**:
   - Handles data manipulation, cleaning, and transformation.
4. **Scikit-learn**:
   - Implements machine learning algorithms, including Random Forest and PCA.
5. **Numpy**:
   - Supports mathematical and numerical operations, especially in scoring and pairwise calculations.
6. **Pickle**:
   - Serializes and deserializes the cleaned dataset and trained machine learning models for faster runtime performance.
7. **CountVectorizer**:
   - Transforms text data into numerical vectors for text-based search.
8. **Cosine Similarity**:
   - Evaluates the similarity between user queries and property descriptions.

---

### Ethical Concerns

1. **Data Integrity**:
   - The dataset must be free of bias or inaccuracies to ensure fair and accurate property evaluations.
   - Cleaning and validation processes mitigate potential data quality issues.
2. **Privacy**:
   - The dataset does not contain personally identifiable information (PII).
3. **Accessibility**:
   - The interface is designed to be intuitive and accessible to a wide range of users.
4. **Prediction Limitations**:
   - Predictions and recommendations are based on historical data and may not reflect future market trends. Clear disclaimers are provided.
5. **Bias in Algorithms**:
   - Machine learning models may inherit biases present in the dataset. Continuous evaluation and retraining can mitigate this risk.

---

### Potential Stakeholders

1. **Renters**:
   - Users looking for apartments that meet their budget and preferences.
2. **Real Estate Agents**:
   - Professionals seeking to streamline property recommendations for clients.
3. **Property Managers**:
   - Use the platform to identify market trends and compare property prices.
4. **Researchers**:
   - Analyze rental trends and user preferences.

---

### Key Features

1. **Filtered Search**:
   - Real-time filtering based on user preferences.
2. **Custom Scoring**:
   - Allows users to rank properties based on their unique priorities.
3. **Price Prediction**:
   - Machine learning model estimates rental prices for listings.
4. **Recommendations**:
   - Provides suggestions for similar apartments based on user-selected top properties.
5. **Interactive Visualizations**:
   - Includes scatter plots for geographical distribution and box plots for price vs. amenities.
