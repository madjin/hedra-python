"""
Enhanced logging utilities for CLI operations with per-job logging.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


class JobLogger:
    """Per-job logger that writes detailed logs to files."""
    
    def __init__(self, job_id: str, console: Console):
        self.job_id = job_id
        self.console = console
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create job-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"job_{job_id}_{timestamp}.log"
        
        # Setup file logger
        self.logger = logging.getLogger(f"hedra.job.{job_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Rich console handler for terminal output
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True
        )
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"🎬 Started job {job_id}")
        self.logger.debug(f"Log file: {self.log_file}")
    
    def log_step(self, step: str, data: Dict[str, Any] = None) -> None:
        """Log a workflow step with optional data."""
        message = f"Step: {step}"
        if data:
            message += f" | Data: {json.dumps(data, indent=2)}"
        self.logger.info(message)
    
    def log_api_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, headers: Dict[str, Any] = None, full_url: str = None) -> None:
        """Log comprehensive API request details."""
        self.logger.info(f"🌐 API REQUEST: {method} {endpoint}")
        if full_url:
            self.logger.info(f"📡 Full URL: {full_url}")
        
        if headers:
            safe_headers = self._sanitize_params(headers)
            self.logger.info(f"📋 Headers: {json.dumps(safe_headers, indent=2)}")
        
        if params:
            # Sanitize sensitive data
            safe_params = self._sanitize_params(params)
            self.logger.info(f"📦 Request Payload: {json.dumps(safe_params, indent=2)}")
    
    def log_api_response(self, response_data: Any, status_code: int = None, headers: Dict[str, Any] = None, truncate_voices: bool = True) -> None:
        """Log comprehensive API response details."""
        if status_code:
            status_emoji = "✅" if 200 <= status_code < 300 else "❌"
            self.logger.info(f"{status_emoji} API Response Status: {status_code}")
        
        if headers:
            safe_headers = self._sanitize_params(headers)
            self.logger.info(f"📋 Response Headers: {json.dumps(safe_headers, indent=2)}")
        
        if hasattr(response_data, 'model_dump'):
            data = response_data.model_dump()
        elif hasattr(response_data, '__dict__'):
            data = response_data.__dict__
        else:
            data = response_data
        
        # Truncate voices list if it's too long
        if truncate_voices and isinstance(data, dict) and 'supported_voices' in data:
            voices = data['supported_voices']
            if len(voices) > 5:
                truncated_data = data.copy()
                truncated_data['supported_voices'] = voices[:3] + [
                    {"...": f"truncated {len(voices) - 3} more voices for brevity"}
                ]
                self.logger.info(f"📨 Response Data (truncated): {json.dumps(truncated_data, indent=2, default=str)}")
                self.logger.info(f"📊 Full voices list: {len(voices)} total voices available")
                return
            
        self.logger.info(f"📨 Response Data: {json.dumps(data, indent=2, default=str)}")
    
    def log_asset_upload(self, asset_type: str, file_path: Path, response_url: str) -> None:
        """Log asset upload with file details."""
        file_size = file_path.stat().st_size / 1024  # KB
        self.logger.info(f"📤 Uploaded {asset_type}: {file_path.name} ({file_size:.1f}KB)")
        self.logger.debug(f"Upload URL: {response_url}")
    
    def log_voice_resolution(self, voice_name: str, voice_id: str) -> None:
        """Log voice name to ID resolution."""
        self.logger.info(f"🔍 Resolved voice: {voice_name} → {voice_id}")
    
    def log_bounding_box(self, bbox_str: str, parsed_coords: Dict[str, float], is_square: bool = None, source: str = "manual") -> None:
        """Log bounding box parsing and coordinates with square verification."""
        self.logger.info(f"🎯 Bounding box: {bbox_str} (source: {source})")
        self.logger.info(f"📍 Parsed coordinates: {json.dumps(parsed_coords, indent=2)}")
        
        if is_square is not None:
            square_emoji = "📦" if is_square else "⚠️"
            self.logger.info(f"{square_emoji} Square bounding box: {is_square}")
            if not is_square:
                self.logger.warning("⚠️  Face bounding boxes should be 1:1 square for optimal animation!")
    
    def log_face_detection(self, image_path: Path, faces_count: int, selected_face: Dict[str, Any] = None) -> None:
        """Log face detection results."""
        self.logger.info(f"👥 Detected {faces_count} faces in {image_path.name}")
        if selected_face:
            self.logger.debug(f"Selected face: {json.dumps(selected_face, indent=2)}")
    
    def log_project_status(self, status: str, progress: float = None, video_url: str = None) -> None:
        """Log project status updates."""
        message = f"📊 Status: {status}"
        if progress is not None:
            message += f" ({progress:.1%})"
        self.logger.info(message)
        
        if video_url:
            self.logger.info(f"🎥 Video ready: {video_url}")
    
    def log_error(self, error: Exception, context: str = None) -> None:
        """Log error details with context."""
        if context:
            self.logger.error(f"❌ Error in {context}: {type(error).__name__}: {error}")
        else:
            self.logger.error(f"❌ Error: {type(error).__name__}: {error}")
        
        # Log full traceback to file only
        import traceback
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
    def log_api_format_selection(self, format_type: str, reasons: list, asset_ids: Dict[str, str] = None) -> None:
        """Log which API format was selected and why."""
        self.logger.info(f"🔧 API Format Selected: {format_type}")
        self.logger.info(f"📋 Selection Reasons: {', '.join(reasons)}")
        
        if asset_ids:
            self.logger.info(f"🏷️  Asset IDs Used:")
            for asset_type, asset_id in asset_ids.items():
                self.logger.info(f"   • {asset_type}: {asset_id}")

    def log_asset_id_extraction(self, url: str, asset_id: str, asset_type: str) -> None:
        """Log asset ID extraction from URLs."""
        self.logger.info(f"🔍 Asset ID Extraction:")
        self.logger.info(f"   📡 Source URL: {url[:100]}{'...' if len(url) > 100 else ''}")
        self.logger.info(f"   🏷️  Extracted ID: {asset_id}")
        self.logger.info(f"   📁 Asset Type: {asset_type}")

    def log_web_app_payload(self, payload: Dict[str, Any]) -> None:
        """Log the complete web-app format payload being sent."""
        self.logger.info(f"🚀 Web-App Format Payload:")
        self.logger.info(f"   • Type: {payload.get('type', 'N/A')}")
        self.logger.info(f"   • AI Model: {payload.get('ai_model_id', 'N/A')}")
        self.logger.info(f"   • Start Keyframe: {payload.get('start_keyframe_id', 'N/A')}")
        self.logger.info(f"   • Audio ID: {payload.get('audio_id', 'N/A')}")
        
        if 'generated_video_inputs' in payload:
            gvi = payload['generated_video_inputs']
            self.logger.info(f"   📋 Video Generation Inputs:")
            for key, value in gvi.items():
                self.logger.info(f"      - {key}: {value}")

    def log_download_details(self, file_path: Path, url: str, file_size: int) -> None:
        """Log detailed download information."""
        size_mb = file_size / (1024 * 1024)
        self.logger.info(f"⬇️  Download Details:")
        self.logger.info(f"   📄 File: {file_path}")
        self.logger.info(f"   📊 Size: {size_mb:.2f}MB ({file_size:,} bytes)")
        self.logger.info(f"   📡 Source: {url[:100]}{'...' if len(url) > 100 else ''}")

    def log_completion(self, success: bool, output_file: Path = None) -> None:
        """Log job completion."""
        if success:
            message = f"✅ Job {self.job_id} completed successfully"
            if output_file:
                message += f" → {output_file}"
            self.logger.info(message)
        else:
            self.logger.error(f"❌ Job {self.job_id} failed")
        
        self.logger.info(f"📋 Full log saved to: {self.log_file}")
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from params for logging."""
        safe_params = params.copy()
        
        # Replace sensitive keys with placeholders
        sensitive_keys = ['api_key', 'token', 'password', 'secret']
        for key in sensitive_keys:
            if key in safe_params:
                safe_params[key] = "[REDACTED]"
        
        # Truncate very long URLs or base64 data
        for key, value in safe_params.items():
            if isinstance(value, str) and len(value) > 200:
                if value.startswith(('http', 'data:')):
                    safe_params[key] = value[:100] + "...[TRUNCATED]"
        
        return safe_params


class EnhancedProgressTracker:
    """Enhanced progress tracker with logging integration."""
    
    def __init__(self, console: Console, job_logger: JobLogger = None):
        self.console = console
        self.job_logger = job_logger
    
    def show_success(self, message: str) -> None:
        """Show success message and log it."""
        self.console.print(f"[green]✅ {message}[/green]")
        if self.job_logger:
            self.job_logger.logger.info(f"✅ {message}")
    
    def show_warning(self, message: str) -> None:
        """Show warning message and log it."""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        if self.job_logger:
            self.job_logger.logger.warning(f"⚠️  {message}")
    
    def show_info(self, message: str) -> None:
        """Show info message and log it."""
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
        if self.job_logger:
            self.job_logger.logger.info(f"ℹ️  {message}")
    
    def show_error(self, message: str) -> None:
        """Show error message and log it."""
        self.console.print(f"[red]❌ {message}[/red]")
        if self.job_logger:
            self.job_logger.logger.error(f"❌ {message}")


def setup_hedra_logging(debug: bool = False) -> None:
    """Setup the main Hedra SDK logging."""
    from hedra._utils._logs import setup_logging
    
    if debug:
        os.environ["HEDRA_LOG"] = "debug"
    else:
        os.environ["HEDRA_LOG"] = "info"
    
    setup_logging()


def create_job_logger(job_id: str, console: Console) -> JobLogger:
    """Create a new job logger instance."""
    return JobLogger(job_id, console)