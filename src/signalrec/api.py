"""The HTTP API (DESIGN.md, "Day 4: serving and experiments"; port 8116).

- ``GET /recommendations/{user}?k=10&exclude_genre=Horror``: the user's top k (MovieLens ids, titles, genres, the
  final score), through ``serving.Recommender``. The ``Server-Timing`` header carries each stage's milliseconds. With
  an experiment configured, the user's arm (``abtest.assign``) decides whether the ranker runs, and the response says
  which arm it was.
- ``POST /events``: ``{"user": 1, "movie": 1193, "rating": 5.0}``. A rating at or above the model's ``min_rating`` is
  a positive (the event script: features, popularity, the rated set); a lower one only joins the rated set. Both are
  idempotent: a retried request changes nothing the second time, and the response says whether it applied.
- ``GET /health``: Redis answers, and which model is served.

Users and movies are MovieLens ids. A user the prepared data doesn't know is a 404 (the store is keyed by the model's
dense user index; new users are a limitation the README states), as is a movie outside the catalogue. Redis being
unreachable is a 503; values in Redis the store didn't write are a 500 naming them.
"""

from __future__ import annotations

from typing import Annotated, Any

import redis
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from signalrec import abtest
from signalrec.features import Event
from signalrec.featurestore import FeatureStoreError
from signalrec.serving import Recommendation, Recommender

MAX_K = 100


class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: StrictInt = Field(ge=1)
    movie: StrictInt = Field(ge=1)
    rating: StrictFloat | StrictInt = Field(ge=0.5, le=5.0)
    time: StrictInt | None = Field(default=None, ge=0)  # Unix seconds; informational (the store keeps arrival order)


def server_timing(rec: Recommendation) -> str:
    return ", ".join(f"{name.replace(' ', '-')};dur={seconds * 1000:.3f}" for name, seconds in rec.seconds.items())


def create_app(recommender: Recommender, experiment: abtest.Experiment | None = None) -> FastAPI:
    model = recommender.model
    store = recommender.store
    min_rating = float(model.info.get("min_rating", 4.0))
    app = FastAPI(title="Signal", version="0.1.0")

    def dense_user(raw: int) -> int:
        try:
            return model.user_index(raw)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from None

    def dense_movie(raw: int) -> int:
        try:
            return model.movie_index(raw)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from None

    def store_errors(exc: Exception) -> HTTPException:
        if isinstance(exc, redis.ConnectionError | redis.TimeoutError):
            return HTTPException(status_code=503, detail=f"feature store unavailable: {exc}")
        return HTTPException(status_code=500, detail=f"feature store: {exc}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            store.client.ping()
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            raise store_errors(exc) from None
        return {
            "status": "ok",
            "movies": model.n_items,
            "users": len(model.user_raw_ids),
            "model": model.info.get("npz_sha256", model.info.get("dataset")),
            "experiment": None if experiment is None else experiment.name,
        }

    @app.get("/recommendations/{user}")
    def recommendations(
        user: int,
        response: Response,
        k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
        exclude_genre: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        dense = dense_user(user)
        arm = None if experiment is None else abtest.assign(experiment, user)
        try:
            rec = recommender.recommend(dense, k=k, exclude_genres=exclude_genre or (), rank=arm != abtest.CONTROL)
        except KeyError as exc:  # an unknown genre
            raise HTTPException(status_code=422, detail=str(exc.args[0])) from None
        except (redis.RedisError, FeatureStoreError) as exc:
            raise store_errors(exc) from None
        response.headers["Server-Timing"] = server_timing(rec)
        items = [
            {
                "movie": int(model.item_raw_ids[m]),
                "title": model.titles[m],
                "genres": [model.catalogue.genre_names[g] for g in model.catalogue.movie_genres[m]],
                "score": float(score),
            }
            for m, score in zip(rec.movies.tolist(), rec.scores.tolist(), strict=True)
        ]
        return {"user": user, "k": k, "arm": arm, "items": items}

    @app.post("/events")
    def events(event: EventIn) -> dict[str, Any]:
        dense = Event(dense_user(event.user), dense_movie(event.movie), event.time or 0)
        positive = float(event.rating) >= min_rating
        try:
            applied = store.record(dense) if positive else store.record_rating(dense)
        except (redis.RedisError, FeatureStoreError) as exc:
            raise store_errors(exc) from None
        return {"applied": applied, "positive": positive}

    return app
