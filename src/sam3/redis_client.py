import redis
from sam3.config import REDIS_HOST, REDIS_PORT

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=False,
    socket_timeout=5,
    ssl=False,
)
