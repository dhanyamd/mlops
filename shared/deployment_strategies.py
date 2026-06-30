import random
import logging
import asyncio
from typing import Callable, Any, Dict, Optional, Awaitable

logger = logging.getLogger(__name__)

class DeploymentRouter:
    """
    Handles traffic routing for different deployment strategies.
    Supported strategies: Shadow, Canary, A/B Testing.
    """
    def __init__(self):
        pass

    async def route_shadow(self, 
                           primary_model: Callable[..., Awaitable[Any]], 
                           shadow_model: Callable[..., Awaitable[Any]], 
                           *args, **kwargs) -> Any:
        """
        Shadow deployment: routes request to primary and shadow concurrently.
        Returns the primary model's result, logging the shadow model's result asynchronously.
        """
        # Await the primary model synchronously (from the user's perspective)
        primary_result = await primary_model(*args, **kwargs)

        # Run shadow model asynchronously so it doesn't block
        async def run_shadow():
            try:
                shadow_result = await shadow_model(*args, **kwargs)
                # In a real system, you'd compare primary_result and shadow_result
                # and emit metrics to Prometheus/Datadog here.
                logger.info("Shadow model prediction completed.", extra={
                    "primary_result": primary_result,
                    "shadow_result": shadow_result
                })
            except Exception as e:
                logger.error(f"Shadow model prediction failed: {str(e)}", exc_info=True)

        asyncio.create_task(run_shadow())
        
        return primary_result

    async def route_canary(self,
                           primary_model: Callable[..., Awaitable[Any]],
                           canary_model: Callable[..., Awaitable[Any]],
                           canary_traffic_percent: int,
                           *args, **kwargs) -> Any:
        """
        Canary deployment: routes a percentage of traffic to the canary model.
        The rest goes to the primary model.
        """
        if random.randint(1, 100) <= canary_traffic_percent:
            logger.info("Routing traffic to Canary model")
            return await canary_model(*args, **kwargs)
        else:
            return await primary_model(*args, **kwargs)

    async def route_ab_test(self,
                            model_a: Callable[..., Awaitable[Any]],
                            model_b: Callable[..., Awaitable[Any]],
                            routing_key: str,
                            *args, **kwargs) -> Any:
        """
        A/B Testing: routes traffic based on a hash of the routing key (e.g. user_id).
        Ensures consistent routing for the same user.
        """
        hash_val = hash(routing_key)
        if hash_val % 2 == 0:
            logger.info(f"A/B Test: Routing {routing_key} to Model A")
            return await model_a(*args, **kwargs)
        else:
            logger.info(f"A/B Test: Routing {routing_key} to Model B")
            return await model_b(*args, **kwargs)
