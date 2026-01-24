"""
YouTube Transcript Extractor Module

This module handles extracting transcripts from YouTube videos.
It supports extracting video IDs from various YouTube URL formats
and fetching transcripts using the YouTube Transcript API.
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
import re


def extract_video_id(url):
    """
    Extract video ID from various YouTube URL formats.
    
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Extracted video ID
        
    Raises:
        ValueError: If URL format is invalid or video ID cannot be extracted
    """
    # Pattern to match various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_transcript(video_id, languages=None):
    """
    Fetch transcript for a YouTube video.
    
    Args:
        video_id (str): YouTube video ID
        languages (list, optional): Preferred languages for transcript (e.g., ['en', 'hi'])
                                   Defaults to None (English)
    
    Returns:
        list: List of transcript dictionaries with 'text', 'start', and 'duration' keys
        
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        # Create API instance
        api = YouTubeTranscriptApi()
        
        # Fetch transcript using the new API
        if languages:
            fetched_transcript = api.fetch(video_id, languages=languages)
        else:
            # Default to English
            fetched_transcript = api.fetch(video_id)
        
        # Convert to raw data (list of dicts)
        return fetched_transcript.to_raw_data()
            
    except TranscriptsDisabled:
        raise Exception(f"Transcripts are disabled for video: {video_id}")
    except NoTranscriptFound as e:
        raise Exception(f"No transcript found for video: {video_id}. Error: {str(e)}")
    except VideoUnavailable:
        raise Exception(f"Video unavailable or private: {video_id}")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")


def format_transcript(transcript_data):
    """
    Convert transcript JSON data to plain text format.
    
    Args:
        transcript_data (list): List of transcript dictionaries
        
    Returns:
        str: Formatted transcript text with timestamps removed
    """
    # Join all text segments with spaces
    text_segments = [entry['text'] for entry in transcript_data]
    
    # Clean up the text
    full_text = ' '.join(text_segments)
    
    # Remove extra whitespaces and newlines
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Add proper paragraph breaks (heuristic: after sentence-ending punctuation followed by capital letter)
    full_text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', full_text)
    
    return full_text.strip()


def get_youtube_text(url, languages=None):
    """
    Complete pipeline to extract and format text from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        languages (list, optional): Preferred languages for transcript
        
    Returns:
        str: Formatted transcript text
        
    Raises:
        ValueError: If URL is invalid
        Exception: If transcript extraction fails
    """
    print(f"Extracting video ID from URL: {url}")
    video_id = extract_video_id(url)
    print(f"Video ID: {video_id}")
    
    print("Fetching transcript...")
    transcript_data = get_transcript(video_id, languages)
    print(f"Transcript retrieved: {len(transcript_data)} segments")
    
    print("Formatting transcript...")
    formatted_text = format_transcript(transcript_data)
    print(f"Text length: {len(formatted_text)} characters")
    
    return formatted_text
