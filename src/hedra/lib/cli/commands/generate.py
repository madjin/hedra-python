"""
Unified generate command for end-to-end character creation.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import click
from rich.table import Table

from hedra.types.character_create_response import CharacterCreateResponse

from ..utils.output import ErrorHandler, ProgressTracker
from ..utils.logging import JobLogger, EnhancedProgressTracker, setup_hedra_logging
# Face detection imports - lazy loaded to avoid TensorFlow overhead
from ..face import get_face_detector, get_face_selector


def extract_asset_id(s3_url: str) -> str | None:
    """Extract asset ID from S3 URL.
    
    S3 URLs contain UUIDs where the second UUID is the asset ID.
    Format: https://s3.amazonaws.com/bucket/{user_id}/{asset_id}/filename
    """
    uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    matches = re.findall(uuid_pattern, s3_url)
    if len(matches) >= 2:
        return matches[1]  # Second UUID is asset ID, first is user ID
    return None


def _determine_api_format(api_format: str, force_v1: bool, force_web_app: bool, has_bounding_box: bool) -> Tuple[str, str, list]:
    """Determine which API format to use and return format, base URL, and reasons.
    
    Returns:
        Tuple of (format_name, base_url, reasons_list)
    """
    # Base URLs for different API formats
    V1_BASE_URL = "https://mercury.dev.dream-ai.com/api"
    WEB_APP_BASE_URL = "https://mercury.dev.dream-ai.com/api"  # Try same base URL with different endpoints
    
    reasons = []
    
    # Handle explicit force flags
    if force_v1:
        reasons.append("--force-v1 flag specified")
        return ("v1", V1_BASE_URL, reasons)
    
    if force_web_app:
        reasons.append("--force-web-app flag specified")
        return ("web-app", WEB_APP_BASE_URL, reasons)
    
    # Handle explicit API format choice
    if api_format == "v1":
        reasons.append("--api-format=v1 specified")
        return ("v1", V1_BASE_URL, reasons)
    
    if api_format == "web-app":
        reasons.append("--api-format=web-app specified")
        return ("web-app", WEB_APP_BASE_URL, reasons)
    
    # Auto-detection logic
    if api_format == "auto":
        if has_bounding_box:
            reasons.append("Bounding box provided - web-app format supports enhanced bounding box features")
            return ("web-app", WEB_APP_BASE_URL, reasons)
        else:
            reasons.append("No bounding box - v1 format sufficient for basic generation")
            return ("v1", V1_BASE_URL, reasons)
    
    # Default fallback
    reasons.append("Default fallback to v1 format")
    return ("v1", V1_BASE_URL, reasons)


@click.command()
@click.option(
    "--text",
    help="Text to convert to speech (required unless using --preview-faces or --label-faces)",
)
@click.option(
    "--voice-id",
    help="Voice ID to use for TTS (e.g., Xb7hH8MSUJpSbSDYk0k2 for Alice)",
)
@click.option(
    "--voice-name",
    help="Voice name to use for TTS (e.g., Alice, Brian)",
)
@click.option(
    "--img",
    "image_file", 
    type=click.Path(exists=True, path_type=Path),
    help="Source image file to animate (e.g., news-interview.png)",
)
@click.option(
    "--avatar-file",
    type=click.Path(exists=True, path_type=Path),
    help="Avatar/portrait reference image (will be cropped and saved to assets/)",
)
@click.option(
    "--avatar-image-url",
    help="URL of already uploaded avatar image",
)
@click.option(
    "--avatar-prompt",
    help="Text prompt for AI-generated avatar (alternative to --img)",
)
@click.option(
    "--avatar-seed",
    type=int,
    help="Seed for AI avatar generation (use with --avatar-prompt)",
)
@click.option(
    "--bounding-box",
    help="Face bounding box in source image as 'x,y,width,height' (normalized 0-1)",
)
@click.option(
    "--aspect-ratio",
    type=click.Choice(["1:1", "16:9", "9:16"]),
    default="16:9",
    help="Video aspect ratio",
)
@click.option(
    "--resolution",
    type=click.Choice(["512", "768", "1024", "720p", "1080p"]),
    help="Video resolution (advanced parameter)",
)
@click.option(
    "--ai-model-id",
    help="AI model ID for generation (advanced parameter)",
)
@click.option(
    "--animation-prompt",
    help="Animation style prompt (advanced parameter)",
)
@click.option(
    "--auto-face-select",
    is_flag=True,
    help="Automatically select best face if multiple detected",
)
@click.option(
    "--select-face",
    is_flag=True,
    help="Interactive face selection with ASCII preview",
)
@click.option(
    "--preview-faces",
    is_flag=True,
    help="Show all detected faces without generation",
)
@click.option(
    "--label-faces",
    is_flag=True,
    help="Interactive face labeling for recognition database",
)
@click.option(
    "--use-face",
    help="Use previously labeled face by name (requires DeepFace)",
)
@click.option(
    "--detector-backend",
    type=click.Choice(["retinaface", "mtcnn", "opencv", "ssd", "dlib"]),
    default="retinaface",
    help="Face detection backend (default: retinaface)",
)
@click.option(
    "--min-face-confidence",
    type=float,
    default=0.85,
    help="Minimum confidence for face detection (default: 0.85)",
)
@click.option(
    "--debug-faces",
    is_flag=True,
    help="Create debug image showing detected and rejected faces",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for generation to complete",
)
@click.option(
    "--download/--no-download", 
    default=True,
    help="Download the video when complete",
)
@click.option(
    "--debug-payload",
    is_flag=True,
    help="Show detailed API payload information for debugging",
)
@click.option(
    "--api-format",
    type=click.Choice(["auto", "v1", "web-app"]),
    default="auto",
    help="Force specific API format: auto (smart detection), v1 (classic), web-app (enhanced)",
)
@click.option(
    "--force-v1",
    is_flag=True,
    help="Force v1 API format (legacy compatibility)",
)
@click.option(
    "--force-web-app",
    is_flag=True,
    help="Force web-app API format (enhanced features)",
)
@click.option(
    "--enable-logging",
    is_flag=True,
    help="Enable detailed per-job logging to logs/ directory",
)
@click.option(
    "--visualize-bbox",
    is_flag=True,
    help="Create a visual overlay showing the bounding box on the image",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output video file path (default: auto-generated)",
)
@click.pass_context
def generate(
    ctx: click.Context,
    text: str,
    voice_id: str | None,
    voice_name: str | None,
    image_file: Path | None,
    avatar_file: Path | None,
    avatar_image_url: str | None,
    avatar_prompt: str | None,
    avatar_seed: int | None,
    bounding_box: str | None,
    aspect_ratio: str,
    resolution: str | None,
    ai_model_id: str | None,
    animation_prompt: str | None,
    auto_face_select: bool,
    select_face: bool,
    preview_faces: bool,
    label_faces: bool,
    use_face: str | None,
    detector_backend: str,
    min_face_confidence: float,
    debug_faces: bool,
    wait: bool,
    download: bool,
    output: Path | None,
    debug_payload: bool,
    api_format: str,
    force_v1: bool,
    force_web_app: bool,
    enable_logging: bool,
    visualize_bbox: bool,
) -> None:
    """🎬 Generate a character video with unified workflow.
    
    This command handles the complete workflow:
    1. Process avatar (crop, save to assets/, store embedding)
    2. Upload source image for animation with bounding box
    3. Resolve voice ID (if voice name provided)
    4. Create character with all parameters
    5. Wait for completion (optional)
    6. Download result (optional)
    
    Examples:
    
        # Correct workflow: separate avatar and source image
        hedra generate \\
          --text "Welcome to AI news" \\
          --voice-name "Alice" \\
          --img "news-interview.png" \\
          --avatar-file "anchor-headshot.jpg" \\
          --bounding-box "0.243,0.277" \\
          --aspect-ratio "16:9" \\
          --ai-model-id "d1dd37a3-e39a-4854-a298-6510289f9cf2" \\
          --animation-prompt "News anchor speaking confidently"
        
        # Legacy mode: use source image as avatar (backward compatibility)
        hedra generate \\
          --text "Hello world" \\
          --voice-name "Alice" \\
          --img "face.jpg"
          
        # AI-generated avatar with source image
        hedra generate \\
          --text "Hello from the future" \\
          --voice-name "Brian" \\
          --img "news-interview.png" \\
          --avatar-prompt "professional news anchor, confident expression" \\
          --bounding-box "0.243,0.277"
    """
    hedra_ctx = ctx.obj
    console = hedra_ctx.console
    error_handler = ErrorHandler(console)
    
    # Setup enhanced logging if requested
    job_logger = None
    if enable_logging:
        setup_hedra_logging(debug=hedra_ctx.debug)
        # Create temporary job_logger with placeholder ID for pre-creation logging
        temp_job_id = str(uuid.uuid4())
        job_logger = JobLogger(temp_job_id, console)
        progress = EnhancedProgressTracker(console, job_logger)
    else:
        progress = ProgressTracker(console)
    
    if not hedra_ctx.client:
        console.print("[red]❌ No API client available[/red]")
        console.print("Set HEDRA_API_KEY environment variable or use --api-key option")
        return
    
    try:
        progress.show_info("🎬 Starting unified character generation workflow")
        
        # Handle face selection modes first (these don't require text)
        if preview_faces and image_file:
            from ..face.face_selector import preview_faces
            preview_faces(str(image_file))
            return
            
        if label_faces and image_file:
            from ..face.face_selector import label_faces_interactive
            labeled_count = label_faces_interactive(str(image_file))
            progress.show_success(f"Labeled {labeled_count} faces for future recognition")
            return
            
        # Validate required text parameter for generation modes
        if not text:
            console.print("[red]❌ --text is required for video generation[/red]")
            console.print("💡 Use --preview-faces or --label-faces to work with faces without generating")
            return
        
        # Handle face-based coordinate selection
        face_coordinates = None
        if use_face and image_file:
            from ..face.face_selector import find_face_by_label
            face_coordinates = find_face_by_label(str(image_file), use_face)
            if not face_coordinates:
                console.print(f"[red]❌ Face '{use_face}' not found. Use --label-faces first.[/red]")
                return
            progress.show_info(f"🧠 Using labeled face: {use_face}")
            
        elif select_face and image_file:
            from ..face.face_selector import select_face
            face_coordinates = select_face(str(image_file), mode='interactive')
            if not face_coordinates:
                console.print("[yellow]Face selection cancelled[/yellow]")
                return
            progress.show_info("🎯 Face selected interactively")
            
        elif auto_face_select and image_file:
            from ..face.face_selector import EnhancedFaceSelector
            try:
                selector = EnhancedFaceSelector(str(image_file))
                # Override detector backend if specified
                if hasattr(selector, 'recognizer') and selector.recognizer:
                    selector.recognizer.detector_backend = detector_backend
                
                face_coordinates = selector.auto_select_best(
                    min_confidence=min_face_confidence,
                    debug_faces=debug_faces
                )
                if not face_coordinates:
                    console.print("[red]❌ No faces detected for auto-selection[/red]")
                    console.print(f"💡 Try lowering --min-face-confidence (current: {min_face_confidence})")
                    return
                progress.show_info("🤖 Best face selected automatically")
            except Exception as e:
                console.print(f"[red]❌ Auto-selection failed: {e}[/red]")
                return
        
        # Convert face coordinates to bounding box if we got coordinates
        if face_coordinates:
            bounding_box = f"{face_coordinates[0]:.3f},{face_coordinates[1]:.3f}"
            progress.show_info(f"📍 Using coordinates: {bounding_box}")
        
        # Debug: Show bounding box visualization if requested
        if visualize_bbox and image_file and bounding_box:
            _visualize_bounding_box(image_file, bounding_box, console, progress)
        
        # Step 1: Validate inputs
        if not voice_id and not voice_name:
            console.print("[red]❌ Either --voice-id or --voice-name is required[/red]")
            return
            
        if not image_file and not avatar_file and not avatar_image_url and not avatar_prompt:
            console.print("[red]❌ One of --img, --avatar-file, --avatar-image-url, or --avatar-prompt is required[/red]")
            return
        
        # Step 2: Determine API format and base URL
        api_format_used, base_url_used, format_reasons = _determine_api_format(
            api_format, force_v1, force_web_app, bounding_box is not None
        )
        
        # Create client with correct base URL
        current_base_url = getattr(hedra_ctx.client, 'base_url', None)
        if base_url_used != current_base_url:
            from hedra import Hedra
            hedra_ctx.client = Hedra(
                api_key=hedra_ctx.config.api_key,
                base_url=base_url_used
            )
            
        if job_logger:
            job_logger.log_api_format_selection(api_format_used, format_reasons)
            job_logger.log_step("API Configuration", {
                "format": api_format_used,
                "base_url": base_url_used,
                "reasons": format_reasons,
                "hybrid_mode": api_format_used == "web-app",
                "note": "Using hybrid approach: v1 for uploads, web-app for generation" if api_format_used == "web-app" else "Using standard v1 API"
            })
            
        progress.show_info(f"🔧 Using {api_format_used} API format")
        
        # Step 3: Resolve voice ID if voice name provided
        resolved_voice_id = voice_id
        if voice_name and not voice_id:
            progress.show_info(f"🔍 Resolving voice name: {voice_name}")
            
            # Use v1 API for voices even if main API is web-app format
            voice_base_url = "https://mercury.dev.dream-ai.com/api"
            if base_url_used != voice_base_url:
                # Temporarily switch to v1 for voice resolution
                from hedra import Hedra
                voice_client = Hedra(
                    api_key=hedra_ctx.config.api_key,
                    base_url=voice_base_url
                )
                if job_logger:
                    job_logger.log_step("Voice Resolution", {
                        "note": "Using v1 API for voice lookup (web-app doesn't support /v1/voices)",
                        "voice_api_base": voice_base_url,
                        "main_api_base": base_url_used
                    })
            else:
                voice_client = hedra_ctx.client
            
            with console.status("[bold green]Fetching voices..."):
                if job_logger:
                    job_logger.log_api_request(
                        "GET", 
                        "/v1/voices",
                        full_url=f"{voice_base_url}/v1/voices",
                        headers={"X-API-Key": "[REDACTED]"}
                    )
                
                voices_response = voice_client.voices.list()
                
                if job_logger:
                    job_logger.log_api_response(voices_response, status_code=200)
            
            for voice in voices_response.supported_voices:
                if voice.name.lower() == voice_name.lower():
                    resolved_voice_id = voice.voice_id
                    progress.show_info(f"✅ Found voice: {voice.name} ({resolved_voice_id})")
                    if job_logger:
                        job_logger.log_voice_resolution(voice_name, resolved_voice_id)
                    break
            else:
                console.print(f"[red]❌ Voice '{voice_name}' not found[/red]")
                console.print("Available voices:")
                for voice in voices_response.supported_voices[:10]:  # Show first 10
                    console.print(f"  - {voice.name}")
                return
        
        # Step 4: Handle avatar/portrait creation
        avatar_image_final_url = avatar_image_url
        
        if avatar_file and not avatar_image_final_url:
            progress.show_info(f"👤 Processing avatar file: {avatar_file.name}")
            
            # Process avatar: crop, save to assets/, store embedding
            avatar_image_final_url = _process_avatar_file(
                avatar_file, console, progress, hedra_ctx.client, aspect_ratio
            )
            
        elif image_file and not avatar_file and not avatar_image_final_url and not avatar_prompt:
            # Legacy mode: use source image as avatar (for backward compatibility)
            progress.show_info(f"📸 Using source image as avatar: {image_file.name}")
            
            upload_file = image_file
            
            # Handle face detection for multi-face images (legacy mode)
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not face_coordinates:
                upload_file = _handle_face_detection(
                    image_file, auto_face_select, console, progress
                )
            
            # Upload the image
            progress.show_info(f"☁️  Uploading avatar image: {upload_file.name}")
            with console.status("[bold green]Uploading avatar image..."):
                # Always use v1 API for portrait uploads (web-app uses different upload flow)
                portrait_base_url = "https://mercury.dev.dream-ai.com/api"
                if base_url_used != portrait_base_url:
                    from hedra import Hedra
                    portrait_client = Hedra(
                        api_key=hedra_ctx.config.api_key,
                        base_url=portrait_base_url
                    )
                    if job_logger:
                        job_logger.log_step("Portrait Upload", {
                            "note": "Using v1 API for portrait upload (web-app uses different upload flow)",
                            "portrait_api_base": portrait_base_url,
                            "main_api_base": base_url_used
                        })
                else:
                    portrait_client = hedra_ctx.client
                
                if job_logger:
                    job_logger.log_api_request(
                        "POST",
                        "/v1/portrait",
                        full_url=f"{portrait_base_url}/v1/portrait?aspect_ratio={aspect_ratio}",
                        headers={"X-API-Key": "[REDACTED]", "Content-Type": "multipart/form-data"},
                        params={"file": f"{upload_file.name} ({upload_file.stat().st_size} bytes)", "aspect_ratio": aspect_ratio}
                    )
                
                portrait_response = portrait_client.portraits.create(
                    file=upload_file,
                    aspect_ratio=aspect_ratio,
                )
                avatar_image_final_url = portrait_response.url
                
                if job_logger:
                    job_logger.log_api_response(portrait_response, status_code=200)
                    job_logger.log_asset_id_extraction(avatar_image_final_url, extract_asset_id(avatar_image_final_url) or "N/A", "avatar_image")
            
            if job_logger:
                job_logger.log_asset_upload("avatar_image", upload_file, avatar_image_final_url)
            
            progress.show_success("Avatar image uploaded!")
            
            # Clean up temporary file if created
            if upload_file != image_file and upload_file.exists():
                upload_file.unlink()
        
        # Step 4: Prepare character creation parameters
        create_params: Dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "audio_source": "tts", 
            "text": text,
            "voice_id": resolved_voice_id,
        }
        
        # Track asset IDs for web app format
        avatar_asset_id = None
        source_asset_id = None
        
        # Add avatar parameters
        if avatar_image_final_url:
            create_params["avatar_image"] = avatar_image_final_url
            # Extract asset ID for web app format
            avatar_asset_id = extract_asset_id(avatar_image_final_url)
            if avatar_asset_id and job_logger:
                job_logger.log_step("Avatar Asset ID Extracted", {
                    "asset_id": avatar_asset_id,
                    "url": avatar_image_final_url
                })
        elif avatar_prompt:
            create_params["avatar_image_input"] = {
                "prompt": avatar_prompt,
            }
            if avatar_seed is not None:
                create_params["avatar_image_input"]["seed"] = avatar_seed
        
        # Step 5: Handle source image and bounding box for animation
        if image_file and bounding_box:
            progress.show_info(f"🎯 Applying bounding box to source image: {image_file.name}")
            
            # Upload source image with bounding box for animation
            source_image_url = _upload_source_image_with_bbox(
                image_file, bounding_box, console, progress, hedra_ctx.client, aspect_ratio,
                job_logger=job_logger, config=hedra_ctx.config
            )
            
            # Extract asset ID for web app format
            source_asset_id = extract_asset_id(source_image_url)
            if source_asset_id and job_logger:
                job_logger.log_step("Source Asset ID Extracted", {
                    "asset_id": source_asset_id,
                    "url": source_image_url
                })
            
            # Use source image as the target for animation
            create_params["avatar_image"] = source_image_url
        
        # Add advanced parameters via extra_body using web app structure
        extra_body = {}
        generated_video_inputs = {}
        
        # Map resolution 
        final_resolution = None
        if resolution:
            if resolution in ["720p", "1080p"]:
                resolution_map = {"720p": "768", "1080p": "1024"}
                final_resolution = resolution_map[resolution]
            else:
                final_resolution = resolution
        
        # Add bounding box in correct web app format
        if bounding_box:
            bbox_parts = bounding_box.split(',')
            if len(bbox_parts) >= 2:
                bbox_coords = [float(bbox_parts[0]), float(bbox_parts[1])]
                generated_video_inputs["bounding_box_target"] = bbox_coords
                progress.show_info(f"🎯 Using bounding box target: {bbox_coords}")
                
                # Check if this is a square bounding box (1:1 ratio)
                # For face coordinates, we expect roughly square regions
                is_square = True  # Assume square for center-point format
                if len(bbox_parts) == 4:
                    # Full bbox format [x, y, width, height]
                    width, height = float(bbox_parts[2]), float(bbox_parts[3])
                    ratio = width / height if height > 0 else 1.0
                    is_square = 0.9 <= ratio <= 1.1  # 10% tolerance for square
                
                if job_logger:
                    job_logger.log_bounding_box(
                        bounding_box, 
                        {
                            "x": bbox_coords[0],
                            "y": bbox_coords[1],
                            "format": "center_point" if len(bbox_parts) == 2 else "full_bbox",
                            "coordinates_type": "normalized",
                            "expected_range": "[0.0-1.0]"
                        },
                        is_square=is_square,
                        source="auto_face_select" if auto_face_select else "manual"
                    )
        
        # Add other video generation parameters
        if final_resolution:
            generated_video_inputs["resolution"] = final_resolution
        if animation_prompt:
            generated_video_inputs["text_prompt"] = animation_prompt
        if aspect_ratio:
            generated_video_inputs["aspect_ratio"] = aspect_ratio
            generated_video_inputs["duration_ms"] = None  # Web app format
        
        # Determine if we should use web app format based on API format selection
        use_web_app_format = (api_format_used == "web-app")
        
        if use_web_app_format:
            progress.show_info("🔄 Using web app format for enhanced bounding box support")
            # Web app format structure
            extra_body["type"] = "video"
            if ai_model_id:
                extra_body["ai_model_id"] = ai_model_id
            
            # Add asset references if available
            if source_asset_id:
                extra_body["start_keyframe_id"] = source_asset_id
                progress.show_info(f"🎯 Using source image asset ID: {source_asset_id}")
            elif avatar_asset_id:
                extra_body["start_keyframe_id"] = avatar_asset_id
                progress.show_info(f"👤 Using avatar asset ID: {avatar_asset_id}")
            
            # For TTS, we'll need to create audio and get its asset ID
            # This would require additional API call - for now, use existing approach
            
            if generated_video_inputs:
                extra_body["generated_video_inputs"] = generated_video_inputs
            
            # Log the complete web-app payload
            if job_logger:
                job_logger.log_web_app_payload(extra_body)
        else:
            # Standard V1 API format
            progress.show_info("🔧 Using V1 format for compatibility")
            
            # In v1 format, some parameters go directly in create_params
            if ai_model_id:
                # V1 format puts these at top level or in extra_body
                extra_body["ai_model_id"] = ai_model_id
            
            # V1 may still support some generated_video_inputs, but without web-app structure
            if generated_video_inputs:
                # Remove web-app specific fields for v1 compatibility
                v1_video_inputs = generated_video_inputs.copy()
                # V1 format may not support all web-app fields
                extra_body["generated_video_inputs"] = v1_video_inputs
            
            # Note: V1 format uses direct parameters, not start_keyframe_id
            if job_logger:
                job_logger.log_step("V1 Format Used", {
                    "uses_avatar_image_url": bool(create_params.get("avatar_image")),
                    "direct_parameters": True,
                    "web_app_features": False
                })
        
        if extra_body:
            create_params["extra_body"] = extra_body
            format_type = "web app" if use_web_app_format else "standard V1"
            progress.show_info(f"🔧 Using {format_type} format with parameters: {list(extra_body.keys())}")
        
        # Debug: Show full payload if requested
        if debug_payload:
            _show_debug_payload(create_params, extra_body, bounding_box, console, use_web_app_format if 'use_web_app_format' in locals() else False)
        
        # Step 5: Create character
        progress.show_info("🎭 Creating character video...")
        
        if job_logger:
            # Log comprehensive API request details
            full_payload = create_params.copy()
            if "extra_body" in full_payload:
                # Merge extra_body into main payload for logging
                extra_body_data = full_payload.pop("extra_body")
                full_payload.update(extra_body_data)
            
            endpoint = "/public/generations" if use_web_app_format else "/v1/characters"
            job_logger.log_api_request(
                "POST", 
                endpoint,
                full_url=f"{base_url_used}{endpoint}",
                headers={"X-API-Key": "[REDACTED]", "Content-Type": "application/json"},
                params=full_payload
            )
        
        with console.status("[bold green]Submitting character creation..."):
            if use_web_app_format:
                try:
                    # Try web-app API first
                    response = _create_character_web_app(
                        create_params, base_url_used, hedra_ctx.config.api_key, job_logger
                    )
                except Exception as e:
                    if "fallback to v1" in str(e):
                        # Web-app API not available, fall back to enhanced v1
                        progress.show_warning("Web-app API not available, using enhanced v1 API")
                        if job_logger:
                            job_logger.log_step("Auto-Fallback", {
                                "from": "web-app",
                                "to": "enhanced-v1",
                                "reason": "web-app endpoint not available",
                                "maintains_features": ["bounding_box", "asset_ids", "advanced_params"]
                            })
                        response: CharacterCreateResponse = hedra_ctx.client.characters.create(
                            **create_params
                        )
                    else:
                        raise
            else:
                response: CharacterCreateResponse = hedra_ctx.client.characters.create(
                    **create_params
                )
        
        job_id = response.job_id
        
        if job_logger:
            if not use_web_app_format:
                # Only log for v1 API (web-app already logged in the function)
                job_logger.log_api_response(response, status_code=200)
            
            # Update job_logger with real job_id for both API formats
            job_logger.job_id = job_id
            job_logger.logger = logging.getLogger(f"hedra.job.{job_id}")
            
            # Create new log file with real job_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_file = job_logger.logs_dir / f"job_{job_id}_{timestamp}.log"
            
            # Update file handler
            for handler in job_logger.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    job_logger.logger.removeHandler(handler)
            
            file_handler = logging.FileHandler(new_log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            job_logger.logger.addHandler(file_handler)
            job_logger.log_file = new_log_file
            # Update progress tracker to use job logger
            progress = EnhancedProgressTracker(console, job_logger)
            job_logger.log_step("Character Creation Initiated", {
                "job_id": job_id,
                "text": text,
                "voice_id": resolved_voice_id,
                "aspect_ratio": aspect_ratio,
                "has_bounding_box": bool(bounding_box),
                "advanced_params": list(extra_body.keys()) if extra_body else []
            })
        
        # Response already logged in the respective API functions
        
        progress.show_success(f"Character creation job submitted!")
        console.print(f"Job ID: [cyan]{job_id}[/cyan]")
        
        # Step 6: Wait for completion (optional)
        if wait:
            progress.show_info("⏳ Waiting for generation to complete...")
            
            video_url = _wait_for_completion(
                hedra_ctx.client, job_id, console, progress, job_logger
            )
            
            if not video_url:
                return  # Failed or timed out
            
            # Step 7: Download video (optional)
            if download:
                output_path = output or Path(f"{job_id}.mp4")
                _download_video(video_url, output_path, console, progress, job_logger)
                
                if job_logger:
                    job_logger.log_completion(True, output_path)
                
        else:
            console.print("\n[dim]Use these commands to track progress:[/dim]")
            console.print(f"  hedra projects get {job_id}")
            console.print(f"  hedra projects wait {job_id}")
            console.print(f"  hedra projects download {job_id}")
            
            if job_logger:
                job_logger.log_completion(True)
        
    except Exception as e:
        if job_logger:
            job_logger.log_error(e, "Character generation workflow")
            job_logger.log_completion(False)
        error_handler.handle_hedra_error(e)


def _apply_bounding_box_crop(
    image_file: Path, 
    bounding_box: str, 
    console, 
    progress
) -> Path:
    """Apply bounding box cropping to image."""
    try:
        import cv2
        import numpy as np
        
        # Parse bounding box "x,y,width,height" (normalized)
        bbox_parts = bounding_box.split(',')
        if len(bbox_parts) == 2:
            # Assume it's "x,y" center point, create square crop
            x_center, y_center = map(float, bbox_parts)
            crop_size = 0.3  # Default crop size
            x = max(0, x_center - crop_size/2)
            y = max(0, y_center - crop_size/2)
            w = min(crop_size, 1 - x)
            h = min(crop_size, 1 - y)
        elif len(bbox_parts) == 4:
            x, y, w, h = map(float, bbox_parts)
        else:
            raise ValueError("Bounding box must be 'x,y' or 'x,y,width,height'")
        
        progress.show_info(f"✂️  Applying bounding box crop: {x:.3f},{y:.3f},{w:.3f},{h:.3f}")
        
        # Load and crop image
        img = cv2.imread(str(image_file))
        if img is None:
            raise ValueError(f"Could not load image: {image_file}")
        
        height, width = img.shape[:2]
        
        # Convert normalized coordinates to pixels
        x_px = int(x * width)
        y_px = int(y * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # Ensure bounds are valid
        x_px = max(0, x_px)
        y_px = max(0, y_px)
        w_px = min(w_px, width - x_px)
        h_px = min(h_px, height - y_px)
        
        # Crop image
        cropped = img[y_px:y_px+h_px, x_px:x_px+w_px]
        
        # Save cropped image
        output_path = image_file.parent / f"{image_file.stem}_cropped{image_file.suffix}"
        cv2.imwrite(str(output_path), cropped)
        
        progress.show_success(f"Image cropped and saved: {output_path.name}")
        return output_path
        
    except Exception as e:
        progress.show_warning(f"Bounding box crop failed: {e}")
        progress.show_info("Continuing with original image...")
        return image_file


def _handle_face_detection(
    image_file: Path,
    auto_face_select: bool,
    console,
    progress
) -> Path:
    """Handle face detection and selection."""
    face_recognizer = get_face_detector()
    if not face_recognizer:
        return image_file
    
    try:
        faces = face_recognizer.detect_faces(str(image_file))
        if len(faces) <= 1:
            return image_file
            
        progress.show_info(f"👥 Detected {len(faces)} faces in image")
        
        if auto_face_select:
            # Select face with highest confidence
            best_face = max(faces, key=lambda f: f.get('confidence', 0))
            face_selector = get_face_selector()
            if face_selector:
                selected_face_path = face_selector.extract_face(
                    str(image_file), best_face
                )
                if selected_face_path:
                    progress.show_info(f"🤖 Auto-selected best face (confidence: {best_face.get('confidence', 0):.2f})")
                    return Path(selected_face_path)
        else:
            progress.show_warning("Multiple faces detected. Use --auto-face-select for automatic selection")
            progress.show_info("Using original image with all faces")
            
    except Exception as e:
        progress.show_warning(f"Face detection failed: {e}")
    
    return image_file


def _wait_for_completion(client, job_id: str, console, progress, job_logger=None, timeout: int = 300) -> str | None:
    """Wait for job completion and return video URL."""
    import time
    
    start_time = time.time()
    interval = 10
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            console.print(f"[yellow]⚠️  Timeout reached ({timeout}s)[/yellow]")
            console.print(f"Use 'hedra projects wait {job_id}' to continue waiting")
            return None
        
        with console.status(f"[bold green]Checking status... ({elapsed:.0f}s)"):
            project = client.projects.retrieve(job_id)
        
        if job_logger:
            job_logger.log_project_status(
                project.status, 
                getattr(project, 'progress', None),
                getattr(project, 'video_url', None)
            )
        
        if project.status == "Completed":
            progress.show_success(f"Generation completed in {elapsed:.0f}s!")
            video_url = getattr(project, 'video_url', None)
            if video_url:
                console.print(f"Video URL: [green]{video_url}[/green]")
                return video_url
            else:
                console.print("[yellow]⚠️  No video URL available[/yellow]")
                return None
                
        elif project.status == "Failed":
            console.print(f"[red]❌ Generation failed[/red]")
            error_msg = getattr(project, 'error_message', None)
            if error_msg:
                console.print(f"Error: {error_msg}")
                if job_logger:
                    job_logger.log_error(Exception(error_msg), "Video generation")
            return None
            
        elif project.status in ["Processing", "Queued", "InProgress"]:
            status_emoji = "⏳" if project.status in ["Processing", "InProgress"] else "📋"
            console.print(f"[blue]{status_emoji} {project.status}... ({elapsed:.0f}s)[/blue]")
            time.sleep(interval)
            
        else:
            console.print(f"[yellow]Unknown status: {project.status}[/yellow]")
            console.print(f"Continuing to wait... Use Ctrl+C to stop")
            time.sleep(interval)


def _process_avatar_file(
    avatar_file: Path,
    console,
    progress, 
    client,
    aspect_ratio: str
) -> str:
    """Process avatar file: detect faces, crop, save to assets/, store embedding, upload."""
    try:
        # Ensure assets directory exists
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        
        progress.show_info(f"🔍 Analyzing avatar file for face detection")
        
        # Use face detection to find best face
        face_recognizer = get_face_detector()
        upload_file = avatar_file
        
        if face_recognizer and avatar_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                faces = face_recognizer.detect_faces(str(avatar_file))
                if faces:
                    # Get best face
                    best_face = max(faces, key=lambda f: f.get('confidence', 0))
                    
                    # Extract and save face to assets/
                    face_selector = get_face_selector()
                    if face_selector:
                        cropped_path = face_selector.extract_face(str(avatar_file), best_face)
                        if cropped_path:
                            # Move to assets directory
                            asset_filename = f"avatar_{avatar_file.stem}_cropped{avatar_file.suffix}"
                            asset_path = assets_dir / asset_filename
                            
                            import shutil
                            shutil.move(cropped_path, asset_path)
                            upload_file = asset_path
                            
                            progress.show_success(f"💾 Avatar saved to: {asset_path}")
                            
                            # TODO: Store embedding in face memory
                            # face_memory = get_face_memory() 
                            # if face_memory:
                            #     embedding = face_recognizer.create_embedding(str(asset_path))
                            #     face_memory.store_face(embedding, str(asset_path), {"type": "avatar"})
                            
            except Exception as e:
                progress.show_warning(f"Face processing failed: {e}")
                progress.show_info("Using original avatar file")
        
        # Upload the avatar file
        progress.show_info(f"☁️  Uploading avatar: {upload_file.name}")
        with console.status("[bold green]Uploading avatar..."):
            portrait_response = client.portraits.create(
                file=upload_file,
                aspect_ratio=aspect_ratio,
            )
        
        progress.show_success("Avatar uploaded and processed!")
        return portrait_response.url
        
    except Exception as e:
        progress.show_warning(f"Avatar processing failed: {e}")
        # Fallback: upload original file
        with console.status("[bold green]Uploading original avatar..."):
            portrait_response = client.portraits.create(
                file=avatar_file,
                aspect_ratio=aspect_ratio,
            )
        return portrait_response.url


def _create_character_web_app(
    create_params: Dict[str, Any], 
    base_url: str, 
    api_key: str, 
    job_logger=None
) -> Any:
    """Create character using web-app API with direct HTTP call."""
    import httpx
    
    # Prepare payload for web-app API
    payload = create_params.copy()
    
    # Web-app API endpoint
    url = f"{base_url}/public/generations"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    if job_logger:
        job_logger.log_api_request(
            "POST", 
            "/public/generations",
            full_url=url,
            headers={"X-API-Key": "[REDACTED]", "Content-Type": "application/json"},
            params=payload
        )
    
    # Make direct HTTP request
    with httpx.Client() as client:
        response = client.post(url, json=payload, headers=headers)
        
        if job_logger:
            job_logger.log_api_response(
                response.json() if response.status_code == 200 else response.text,
                status_code=response.status_code
            )
        
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error') if response.status_code != 404 else 'Not Found'
            # Web-app API not available, fall back to v1 with enhanced parameters
            if job_logger:
                job_logger.log_step("Web-App Fallback", {
                    "web_app_status": response.status_code,
                    "web_app_error": error_detail,
                    "fallback_action": "Using v1 API with enhanced bounding box parameters",
                    "note": "Web-app endpoint not available, using best v1 features"
                })
            raise Exception(f"Web-app API not available ({response.status_code}: {error_detail}), falling back to v1")
        
        # Convert response to compatible format
        response_data = response.json()
        
        # Create a simple object with job_id attribute for compatibility
        class WebAppResponse:
            def __init__(self, data):
                # Web-app API might return different field names
                self.job_id = data.get('job_id') or data.get('id') or data.get('generation_id')
                self.raw_data = data
                
        return WebAppResponse(response_data)


def _upload_source_image_with_bbox(
    source_image: Path,
    bounding_box: str,
    console,
    progress,
    client,
    aspect_ratio: str,
    job_logger=None,
    config=None
) -> str:
    """Upload source image for animation, with bounding box info preserved."""
    try:
        # For now, we'll upload the source image as-is
        # The bounding box will be handled by the API during animation
        progress.show_info(f"📤 Uploading source image for animation: {source_image.name}")
        progress.show_info(f"📍 Bounding box coordinates: {bounding_box}")
        
        # Always use v1 API for portrait uploads
        portrait_base_url = "https://mercury.dev.dream-ai.com/api"
        current_base_url = getattr(client, 'base_url', None)
        
        if current_base_url != portrait_base_url and config:
            from hedra import Hedra
            portrait_client = Hedra(
                api_key=config.api_key,
                base_url=portrait_base_url
            )
            if job_logger:
                job_logger.log_step("Source Image Upload", {
                    "note": "Using v1 API for source image upload (web-app uses different upload flow)",
                    "portrait_api_base": portrait_base_url,
                    "main_api_base": str(current_base_url) if current_base_url else None
                })
        else:
            portrait_client = client
        
        with console.status("[bold green]Uploading source image..."):
            if job_logger:
                job_logger.log_api_request(
                    "POST",
                    "/v1/portrait",
                    full_url=f"{portrait_base_url}/v1/portrait?aspect_ratio={aspect_ratio}",
                    headers={"X-API-Key": "[REDACTED]", "Content-Type": "multipart/form-data"},
                    params={"file": f"{source_image.name} ({source_image.stat().st_size} bytes)", "aspect_ratio": aspect_ratio}
                )
            
            portrait_response = portrait_client.portraits.create(
                file=source_image,
                aspect_ratio=aspect_ratio,
            )
            
            if job_logger:
                job_logger.log_api_response(portrait_response, status_code=200)
                job_logger.log_asset_id_extraction(portrait_response.url, extract_asset_id(portrait_response.url) or "N/A", "source_image")
        
        progress.show_success("Source image uploaded for animation!")
        
        # NOTE: The bounding box coordinates should be passed to the character creation
        # as metadata or extra parameters - this depends on how Hedra API handles it
        
        return portrait_response.url
        
    except Exception as e:
        progress.show_warning(f"Source image upload failed: {e}")
        raise


def _visualize_bounding_box(
    image_file: Path,
    bounding_box: str,
    console,
    progress
) -> None:
    """Create a visual overlay showing the bounding box on the image."""
    try:
        import cv2
        import numpy as np
        
        progress.show_info(f"🎯 Creating bounding box visualization")
        
        # Parse bounding box
        bbox_parts = bounding_box.split(',')
        if len(bbox_parts) == 2:
            # Center point format: convert to bounding box
            x_center, y_center = map(float, bbox_parts)
            # Create a reasonable crop size around the center
            crop_size = 0.2
            x = max(0, x_center - crop_size/2)
            y = max(0, y_center - crop_size/2)
            w = min(crop_size, 1 - x)
            h = min(crop_size, 1 - y)
        elif len(bbox_parts) == 4:
            x, y, w, h = map(float, bbox_parts)
        else:
            progress.show_warning("Invalid bounding box format")
            return
        
        # Load image
        img = cv2.imread(str(image_file))
        if img is None:
            progress.show_warning(f"Could not load image: {image_file}")
            return
        
        height, width = img.shape[:2]
        
        # Convert normalized coordinates to pixels
        x_px = int(x * width)
        y_px = int(y * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # Draw bounding box
        cv2.rectangle(img, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 3)
        
        # Add center point
        center_x = x_px + w_px // 2
        center_y = y_px + h_px // 2
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add text overlay
        cv2.putText(img, f"BBox: {x:.3f},{y:.3f},{w:.3f},{h:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Center: {x_center:.3f},{y_center:.3f}" if len(bbox_parts) == 2 else f"Size: {w_px}x{h_px}px", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        output_path = image_file.parent / f"{image_file.stem}_bbox_debug{image_file.suffix}"
        cv2.imwrite(str(output_path), img)
        
        progress.show_success(f"Bounding box visualization saved: {output_path}")
        
        # Show detailed info
        console.print("\n📊 [bold blue]Bounding Box Analysis:[/bold blue]")
        console.print(f"  📐 Normalized coordinates: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
        console.print(f"  📏 Pixel coordinates: x={x_px}, y={y_px}, w={w_px}, h={h_px}")
        console.print(f"  🖼️  Image dimensions: {width}x{height}")
        console.print(f"  🎯 Center point: ({center_x}, {center_y})")
        console.print(f"  💾 Visualization: {output_path}")
        
    except Exception as e:
        progress.show_warning(f"Bounding box visualization failed: {e}")


def _show_debug_payload(
    create_params: Dict[str, Any],
    extra_body: Dict[str, Any],
    bounding_box: str | None,
    console,
    use_web_app_format: bool = False
) -> None:
    """Show detailed debug information about the API payload."""
    console.print("\n🐛 [bold blue]DEBUG: API Payload Information[/bold blue]")
    console.print("=" * 60)
    
    # Show format type
    format_type = "Web App Format" if use_web_app_format else "Standard V1 Format"
    console.print(f"\n📋 [bold]Format Type:[/bold] [cyan]{format_type}[/cyan]")
    if use_web_app_format:
        console.print("  🔄 Enhanced bounding box support with asset references")
    else:
        console.print("  🔧 Standard API format")
    
    # Show main parameters
    console.print("\n📋 [bold]Main Parameters:[/bold]")
    for key, value in create_params.items():
        if key != "extra_body":
            if isinstance(value, str) and len(value) > 50:
                display_value = f"{value[:47]}..."
            else:
                display_value = value
            console.print(f"  {key}: [green]{display_value}[/green]")
    
    # Show extra_body (advanced parameters)
    if extra_body:
        console.print("\n🔧 [bold]Advanced Parameters (extra_body):[/bold]")
        for key, value in extra_body.items():
            console.print(f"  {key}: [yellow]{value}[/yellow]")
        
        # Show actual extra_body structure being sent
        console.print("\n📄 [bold]Actual extra_body Structure:[/bold]")
        console.print(json.dumps(extra_body, indent=2))
        
        # Show asset ID info if present
        if "start_keyframe_id" in extra_body:
            console.print(f"\n🎯 [bold]Asset References:[/bold]")
            console.print(f"  Start Keyframe ID: [green]{extra_body['start_keyframe_id']}[/green]")
        if "audio_id" in extra_body:
            console.print(f"  Audio ID: [green]{extra_body['audio_id']}[/green]")
    
    # Show bounding box info
    if bounding_box:
        console.print(f"\n🎯 [bold]Bounding Box:[/bold]")
        bbox_parts = bounding_box.split(',')
        if len(bbox_parts) == 2:
            console.print(f"  Format: Center point")
            console.print(f"  X Center: [cyan]{bbox_parts[0]}[/cyan]")
            console.print(f"  Y Center: [cyan]{bbox_parts[1]}[/cyan]")
        elif len(bbox_parts) == 4:
            console.print(f"  Format: Full bounding box")
            console.print(f"  X: [cyan]{bbox_parts[0]}[/cyan], Y: [cyan]{bbox_parts[1]}[/cyan]")
            console.print(f"  Width: [cyan]{bbox_parts[2]}[/cyan], Height: [cyan]{bbox_parts[3]}[/cyan]")
    
    console.print("\n" + "=" * 60)


def _download_video(video_url: str, output_path: Path, console, progress, job_logger=None) -> None:
    """Download video from URL."""
    try:
        import httpx
        
        progress.show_info(f"⬇️  Downloading video to: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with console.status(f"[bold green]Downloading..."):
            response = httpx.get(video_url, follow_redirects=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
        
        file_size = output_path.stat().st_size / 1024 / 1024
        progress.show_success(f"Video downloaded: {output_path} ({file_size:.1f}MB)")
        
        if job_logger:
            job_logger.log_step("Video Downloaded", {
                "file_path": str(output_path),
                "file_size_mb": f"{file_size:.1f}MB",
                "source_url": video_url
            })
        
    except Exception as e:
        progress.show_warning(f"Download failed: {e}")
        console.print(f"Manual download: [blue]{video_url}[/blue]")
        if job_logger:
            job_logger.log_error(e, "Video download")