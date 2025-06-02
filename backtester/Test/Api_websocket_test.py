import asyncio
import aiohttp
import logging
from datetime import datetime
from betfairlightweight import StreamListener, BetfairClient

# --- Configuration ---
THE_ODDS_API_KEY = "YOUR_ODDS_API_KEY"
BETFAIR_APP_KEY = "YOUR_BETFAIR_APP_KEY"
USERNAME = "YOUR_BETFAIR_USERNAME"
PASSWORD = "YOUR_BETFAIR_PASSWORD"
CERT_FILE = "certs/client-2048.crt"
KEY_FILE = "certs/client-2048.key"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConnectivityTest")

# --- Odds API Test ---
async def test_odds_api():
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?regions=eu&markets=h2h&apiKey={THE_ODDS_API_KEY}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"✅ Odds API Success: Fetched {len(data)} markets")
                else:
                    logger.warning(f"❌ Odds API Failed: HTTP {resp.status}")
    except Exception as e:
        logger.error(f"❌ Odds API Exception: {e}", exc_info=True)

# --- Betfair WebSocket Test ---
class TestListener(StreamListener):
    def on_data(self, raw_data):
        try:
            for market in raw_data.get("mc", []):
                market_id = market.get("id")
                logger.info(f"📡 Market Update: {market_id}")
        except Exception as e:
            logger.error(f"❌ Stream Processing Error: {e}", exc_info=True)

async def test_betfair_ws():
    try:
        client = BetfairClient(
            app_key=BETFAIR_APP_KEY,
            cert_files=(CERT_FILE, KEY_FILE)
        )
        await client.login(USERNAME, PASSWORD)
        logger.info("✅ Betfair Logged in successfully.")

        listener = TestListener()
        stream = await client.streaming.create_stream(listener=listener, unique_id=100)
        await stream.subscribe_to_markets(
            filter={
                "eventTypeIds": ["1"],
                "marketStartTime": {"from": datetime.utcnow().isoformat()},
                "marketCountries": ["GB"],
                "marketTypeCodes": ["MATCH_ODDS"]
            },
            max_markets=1
        )

        logger.info("📶 Listening for 10 seconds...")
        await asyncio.sleep(10)
        await client.logout()
    except Exception as e:
        logger.error(f"❌ Betfair WS Exception: {e}", exc_info=True)

# --- Run Both Tests ---
async def main():
    await test_odds_api()
    await test_betfair_ws()

if __name__ == "__main__":
    asyncio.run(main())
