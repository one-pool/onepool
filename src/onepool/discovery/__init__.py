"""Zero-config pool discovery on the local network."""

from onepool.discovery.mdns import PoolAdvertisement, find_pool

__all__ = ["PoolAdvertisement", "find_pool"]
