"""FastAPI app for the Yu-Gi-Oh! OpenEnv server."""

from openenv.core.env_server.http_server import create_app

from yugioh_env.models import YuGiOhAction, YuGiOhObservation
from yugioh_env.server.yugioh_environment import YuGiOhEnvironment

app = create_app(
    YuGiOhEnvironment,
    YuGiOhAction,
    YuGiOhObservation,
    env_name="yugioh-env",
)
