# 🎞️ MovieIQ

A Streamlit app that estimates whether a movie is likely to earn back its
budget, trained on budget, popularity, runtime, audience rating, and genre.

## Project structure

```
MovieIQ/
├── app.py                     # Streamlit app (Home, Dashboard, Prediction, Dataset, About)
├── train_model.py             # Trains and saves the Random Forest + genre encoder
├── data_cleaning.ipynb         # Raw data -> Clean Dataset/movies_clean.csv
├── eda.ipynb                  # Exploratory analysis notebook
├── requirements.txt
├── Clean Dataset/
│   └── movies_clean.csv
└── Models/
    ├── movie_success_model.pkl
    └── genre_encoder.pkl
```

## Setup

```bash
pip install -r requirements.txt
python train_model.py   # generates Models/*.pkl (skip if already present)
streamlit run app.py
```

## Pipeline

1. **Clean** — `data_cleaning.ipynb` parses genre data out of a raw JSON-style
   column, drops junk, and flags zero/negative budget, revenue, and runtime.
2. **Engineer** — derives `success` (`revenue > budget`), `profit`, and `roi`.
3. **Explore** — `eda.ipynb` reviews distributions and correlations.
4. **Train** — `train_model.py` fits a Random Forest on
   `[budget, popularity, runtime, vote_average, primary_genre]`.
5. **Serve** — `app.py` loads the saved model and encoder for live scoring.

## Honest limits (read before trusting the numbers)

- **Class imbalance**: ~81% of titles in the dataset are labeled a success.
  A model that always predicts "hit" already scores ~81% accuracy without
  learning anything — that's what a plain Random Forest here does.
- **This build** uses `class_weight="balanced"` and a capped tree depth so
  it actually differentiates, but recall on flops is still weak (~10%):
  these five features carry very little signal for this outcome definition
  in this dataset. Predictions barely move across very different inputs —
  don't read them as precise.
- **Missing genres**: ~9% of rows had no genre after cleaning. They're coded
  as their own `Unknown` category (both in training and at prediction time)
  rather than dropped or silently mismatched.
- Budget, popularity, and rating are treated as known inputs here. In an
  actual pre-release scenario several of these would be forecasts, not facts.

Treat the app as a demonstration of an end-to-end ML pipeline — cleaning,
feature engineering, training, and serving — rather than a production-grade
box-office predictor.

## Author

Shubham Samarpit — Data Analytics
