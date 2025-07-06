#!/usr/bin/env python3
"""
Test using web app format with asset IDs in V1 API
"""

import re
from hedra import Hedra
from pathlib import Path
import json

def extract_asset_id(s3_url):
    """Extract asset ID from S3 URL"""
    uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    matches = re.findall(uuid_pattern, s3_url)
    if len(matches) >= 2:
        return matches[1]  # Second UUID is asset ID, first is user ID
    return None

def test_web_app_format():
    """Test using web app payload format with asset IDs"""
    client = Hedra()
    
    print("🧪 Testing Web App Format with Asset IDs")
    print("=" * 60)
    
    # Step 1: Upload image and extract asset ID
    print("\n1. Uploading image and extracting asset ID...")
    with open('trash/news-interview.png', 'rb') as f:
        portrait_response = client.portraits.create(file=f, aspect_ratio='16:9')
    
    image_asset_id = extract_asset_id(portrait_response.url)
    print(f"   Image Asset ID: {image_asset_id}")
    print(f"   Image URL: {portrait_response.url}")
    
    # Step 2: Create minimal audio and extract asset ID  
    print("\n2. Creating test audio and extracting asset ID...")
    import wave
    import numpy as np
    
    audio_file = Path('temp_audio.wav')
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # Quieter
    audio_data = (audio_data * 16383).astype(np.int16)
    
    with wave.open(str(audio_file), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    audio_response = client.audio.create(file=audio_file)
    audio_asset_id = extract_asset_id(audio_response.url)
    print(f"   Audio Asset ID: {audio_asset_id}")
    print(f"   Audio URL: {audio_response.url}")
    
    # Clean up temp file
    audio_file.unlink()
    
    # Step 3: Test different payload formats
    bounding_box = [0.2798265763408341, 0.34493141374072084]
    
    print("\n3. Testing V1 API Standard Format...")
    v1_params = {
        "aspect_ratio": "16:9",
        "audio_source": "audio",
        "voice_url": audio_response.url,
        "avatar_image": portrait_response.url,
        "extra_body": {
            "ai_model_id": "d1dd37a3-e39a-4854-a298-6510289f9cf2",
            "type": "video",
            "generated_video_inputs": {
                "bounding_box_target": bounding_box,
                "resolution": 768,
                "text_prompt": "News anchor speaking confidently",
                "aspect_ratio": "16:9"
            }
        }
    }
    
    print("V1 Standard Payload:")
    print(json.dumps(v1_params, indent=2))
    
    try:
        response1 = client.characters.create(**v1_params)
        print(f"✅ V1 Standard Format Success - Job ID: {response1.job_id}")
    except Exception as e:
        print(f"❌ V1 Standard Format Failed: {e}")
    
    print("\n4. Testing Web App Format in extra_body...")
    web_app_params = {
        "aspect_ratio": "16:9",
        "audio_source": "audio", 
        "voice_url": audio_response.url,
        "avatar_image": portrait_response.url,
        "extra_body": {
            # Web app format structure
            "type": "video",
            "ai_model_id": "d1dd37a3-e39a-4854-a298-6510289f9cf2",
            "start_keyframe_id": image_asset_id,
            "audio_id": audio_asset_id,
            "generated_video_inputs": {
                "text_prompt": "News anchor speaking confidently",
                "resolution": "768",
                "aspect_ratio": "16:9",
                "duration_ms": None,
                "bounding_box_target": bounding_box
            }
        }
    }
    
    print("Web App Format Payload:")
    print(json.dumps(web_app_params, indent=2))
    
    try:
        response2 = client.characters.create(**web_app_params)
        print(f"✅ Web App Format Success - Job ID: {response2.job_id}")
    except Exception as e:
        print(f"❌ Web App Format Failed: {e}")
    
    print("\n5. Testing Hybrid Format...")
    hybrid_params = {
        "aspect_ratio": "16:9",
        "audio_source": "audio",
        "voice_url": audio_response.url, 
        "avatar_image": portrait_response.url,
        "extra_body": {
            # Mix both approaches
            "ai_model_id": "d1dd37a3-e39a-4854-a298-6510289f9cf2",
            "type": "video",
            "start_keyframe_id": image_asset_id,  # Add asset references
            "audio_id": audio_asset_id,
            "generated_video_inputs": {
                "bounding_box_target": bounding_box,
                "resolution": 768,
                "text_prompt": "News anchor speaking confidently", 
                "aspect_ratio": "16:9",
                "duration_ms": None
            }
        }
    }
    
    print("Hybrid Format Payload:")
    print(json.dumps(hybrid_params, indent=2))
    
    try:
        response3 = client.characters.create(**hybrid_params)
        print(f"✅ Hybrid Format Success - Job ID: {response3.job_id}")
        return response3.job_id
    except Exception as e:
        print(f"❌ Hybrid Format Failed: {e}")
        return None

if __name__ == "__main__":
    job_id = test_web_app_format()
    if job_id:
        print(f"\n🎯 Test job created: {job_id}")
        print("Monitor with: python -m hedra.lib.cli projects get", job_id)