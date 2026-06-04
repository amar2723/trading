from __future__ import annotations

import MetaTrader5 as mt5


def connect() -> bool:
    """Initialize MetaTrader 5 and report whether the terminal is reachable."""
    if not mt5.initialize():
        print(f"Failed: {mt5.last_error()}")
        return False

    print("Connected")
    return True


def shutdown() -> None:
    mt5.shutdown()


if __name__ == "__main__":
    if connect():
        shutdown()
