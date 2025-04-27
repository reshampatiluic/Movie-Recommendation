# kafka_cleaned_stream_processor.py
import re

def clean_raw_message(raw_message):
    try:
        parts = raw_message.strip().split(',')
        timestamp = parts[0]
        user_id = int(parts[1])
        request = parts[2]

        # Parse Movie Name and Minute Watched
        match = re.search(r'/data/m/(.*?)/(\d+)\.mpg', request)
        if match:
            movie_name_raw = match.group(1)
            minute = int(match.group(2))
            movie_name = movie_name_raw.replace('+', ' ').replace('_', ' ')
            return {
                'timestamp': timestamp,
                'user_id': user_id,
                'movie_name': movie_name.lower(),
                'minute': minute
            }
        else:
            # Not a movie watch event
            return None

    except Exception as e:
        print(f"❌ Failed to clean message: {e}")
        return None
