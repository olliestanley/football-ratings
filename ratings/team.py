import math
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RatingSystem():
    """
    Elo-like rating system for teams based on match overview data.

    Args:
        k: Parameter determining recency sensitivity when updating team ratings. Higher
            `k` leads to greater sensitivity of ratings to recent results.
        baseline: Parameter scaling the total output in a match. If using goals or
            expected goals data as the performance metric, a value around 2.7 - 2.9 is
            recommended as this tends to be the number of goals per match.
        home_advantage: Parameter scaling the `baseline` to provide a slight advantage
            to home teams and a slight disadvantage to away teams. Note that the total
            advantage is therefore effectively `(2 * home_advantage)`.
        default_rating: The default rating for new teams. This will also tend to be the
            average rating.
        date_column: Name of column giving the date of matches.
        home_id_column: Name of column giving the unique ID for home team.
        away_id_column: Name of column giving the unique ID for away team.
        home_true_column: Name of column giving true performance metric for home team.
        away_true_column: Name of column giving true performance metric for away team.
        home_score_column: Name of column giving true score for home team.
        away_score_column: Name of column giving true score for away team.
    """

    def __init__(
        self,
        k: int = 32,
        baseline: float = 2.8,
        home_advantage: float = 0.05,
        default_rating: float = 1000,
        date_column: str = "date",
        home_id_column: str = "home_team",
        away_id_column: str = "away_team",
        home_true_column: str = "home_xg",
        away_true_column: str = "away_xg",
        home_score_column: str = "home_goals",
        away_score_column: str = "away_goals",
    ):
        self.k = k
        self.baseline = baseline
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.date_column = date_column
        self.home_id_column = home_id_column
        self.away_id_column = away_id_column
        self.home_true_column = home_true_column
        self.away_true_column = away_true_column
        self.home_score_column = home_score_column
        self.away_score_column = away_score_column

    def process_dataset(self, data: pd.DataFrame) -> None:
        """
        Iterate the given match dataset, developing attacking and defending ratings for
        each team as well as populating each match row with a prior prediction of home
        and away performance in the match.

        This adds 6 new columns to `data`: home_pred, away_pred, home_att_rating,
        home_def_rating, away_att_rating, away_def_rating.
        """

        columns = [
            "home_pred",
            "away_pred",
            "home_att_rating",
            "home_def_rating",
            "away_att_rating",
            "away_def_rating",
        ]

        data[columns] = [
            0,
            0,
            self.default_rating,
            self.default_rating,
            self.default_rating,
            self.default_rating,
        ]

        for index in range(len(data)):
            data.loc[index, columns] = self.process_match(data, index)

    def process_match(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """
        Calculate home and away performance predictions for the match at the given
        `index` in `data`, then compare these to true performance to calculate new
        ratings for each team.

        The `data` should not contain any matches which are not yet played, as ratings
        cannot be updated without true performance data. To predict performance from an
        unplayed match, use `predict_match_from_ratings`.
        """

        row = data.iloc[index]

        home_att, home_def = self.get_team_ratings_before_date(
            data, row[self.home_id_column], row[self.date_column]
        )

        away_att, away_def = self.get_team_ratings_before_date(
            data, row[self.away_id_column], row[self.date_column]
        )

        home_pred, away_pred = self.predict_match_from_ratings(
            home_att, home_def, away_att, away_def
        )

        home_true, away_true = row[[self.home_true_column, self.away_true_column]]

        home_delta, away_delta = home_true - home_pred, away_true - away_pred

        return (
            home_pred,
            away_pred,
            home_att + (home_delta * self.k),
            home_def - (away_delta * self.k),
            away_att + (away_delta * self.k),
            away_def - (home_delta * self.k),
        )

    def fit_forecast_model(self, data: pd.DataFrame) -> None:
        """
        Fit a multiclass logistic regression model forecasting home win, draw, away win
        probabilities from predicted performance values as returned by `process_match`.
        """

        data["outcome"] = np.select(
            condlist=[
                data[self.home_score_column] > data[self.away_score_column],
                data[self.home_score_column] < data[self.away_score_column],
            ],
            choicelist=[0, 2],
            default=1,
        )

        x = data[["home_pred", "away_pred"]]
        y = data["outcome"]

        self.forecast_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", penalty="l2", C=0.1
        ).fit(x, y)

    def forecast_dataset(self, data: pd.DataFrame) -> None:
        """
        Use a fitted multiclass forecasting model to forecast home win, draw, away win
        probabilities for each match in the dataset, based on predicted performance
        values such as from `process_match`. The `data` must have the below columns:
        `home_pred, away_pred`.

        To forecast probabilities using custom predicted performance numbers, such as
        a match which has not yet been played, use `forecast_match_from_predictions`.
        """

        x = data[["home_pred", "away_pred"]]
        y_pred = self.forecast_model.predict_proba(x)
        data[["home_prob", "draw_prob", "away_prob"]] = y_pred

    def forecast_match_from_predictions(self, home_pred, away_pred) -> np.ndarray:
        """
        Use a fitted multiclass forecasting model to forecast home win, draw, away win
        probabilities for a match based on home and away performance predictions.

        Use `predict_match_from_ratings` to obtain performance predictions from a set
        of team ratings.
        """

        return self.forecast_model.predict_proba([[home_pred, away_pred]])

    def predict_match_from_ratings(
        self, home_att: float, home_def: float, away_att: float, away_def: float
    ) -> Tuple[float, float]:
        """
        Predict performance ratings for both teams in a match based on the attacking
        and defending ratings for the teams.

        The predicted ratings are dependent on the parameters of the rating system.
        """

        home_factor = self.baseline * (1 + self.home_advantage)
        away_factor = self.baseline * (1 - self.home_advantage)

        home_pred = home_factor / (1 + math.exp((away_def - home_att) / 400))
        away_pred = away_factor / (1 + math.exp((home_def - away_att) / 400))

        return home_pred, away_pred

    def get_team_ratings_before_date(
        self, data: pd.DataFrame, team: str, date: str
    ) -> float:
        """
        In `data`, get the most recent attacking and defending ratings for `team` which
        was prior to `date`.
        """

        data = data[data[self.date_column] < date]

        match_home = data[data[self.home_id_column] == team]
        home = match_home.iloc[-1] if len(match_home) else None

        match_away = data[data[self.away_id_column] == team]
        away = match_away.iloc[-1] if len(match_away) else None

        if (
            home is not None and 
            (away is None or home[self.date_column] > away[self.date_column])
        ):
            return home["home_att_rating"], home["home_def_rating"]

        if away is not None:
            return away["away_att_rating"], away["away_def_rating"]

        return self.default_rating, self.default_rating
