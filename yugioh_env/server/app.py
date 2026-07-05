"""FastAPI app for the Yu-Gi-Oh! OpenEnv server."""

from fastapi.middleware.cors import CORSMiddleware
from openenv.core.env_server.http_server import create_app

from yugioh_env.models import YuGiOhAction, YuGiOhObservation
from yugioh_env.server.web_api import (
    create_action_describer,
    create_card_text_resolver,
    create_event_describer,
    create_web_env,
    web_router,
)
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

app = create_app(
    YuGiOhEnvironment,
    YuGiOhAction,
    YuGiOhObservation,
    env_name="yugioh-env",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.web_env = create_web_env({"deck_path": "assets/decks/blue_eyes.ydk"})
app.state.action_describer = create_action_describer(app.state.web_env)
app.state.event_describer = create_event_describer(app.state.web_env)
app.state.card_text_resolver = create_card_text_resolver(app.state.web_env)
app.include_router(web_router)
