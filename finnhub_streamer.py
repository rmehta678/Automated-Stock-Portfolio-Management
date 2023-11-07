import finnhub
import datetime
import time

finnhub_client = finnhub.Client(api_key="cl056c1r01qhjei33va0cl056c1r01qhjei33vag")

# Keep looping until the user presses a button
while True:

    # Get the current UTC time
    end_utc = datetime.datetime.utcnow()

    print(datetime.datetime.fromtimestamp(int(end_utc.timestamp())))

    start_utc = end_utc - datetime.timedelta(minutes=500)

    print(datetime.datetime.fromtimestamp(int(start_utc.timestamp())))

    # Get the stock candles for AAPL for the past minute
    candles = finnhub_client.stock_candles('AAPL', 1, int(start_utc.timestamp()), int(end_utc.timestamp()))

    # Print the candles
    print(len(candles['c']))

    # Wait for a minute
    time.sleep(60)

    # Update the start utc time to one minute behind the current UTC time
    start_utc = datetime.datetime.utcnow()

    # If the user presses a button, break out of the loop
    if input('Press any key to stop: ') is not None:
        break